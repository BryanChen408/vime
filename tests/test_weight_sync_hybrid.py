"""混合架构权重同步的离线回归测试。

覆盖 2026-08-21 混合首跑排掉的四颗雷,不烧集群:
1. 扩展类与 NPUWorker 原生方法的属性冲突(vllm worker_base 断言,run 152554)。
2. 共卡/远程引擎切分(8 IPC + 6 HCCL,UpdateWeightFromTensor 混合分发)。
3. 引擎级 offload 过滤(共卡 sleep、专用常驻)。
4. IPC 借用显存 × layerwise 缓冲的生命周期(run 155209 共享专家校验炸点)
   —— 用"源 tensor 交付后被覆写"确定性模拟 trainer 复用显存,验证 clone 修复。

运行: pytest tests/test_weight_sync_hybrid.py -v
"""

import os
import types
import unittest.mock as mock
from contextlib import nullcontext

import pytest
import torch
from torch import nn

from vime.backends.megatron_utils.update_weight.update_weight_from_tensor import (
    count_colocated_engines,
    vLLMColocateWorkerExtension,
)


# ─── 1. 扩展类与 NPUWorker 零冲突(vllm worker_base.py:271 断言扫描) ─────────────
def _class_method_names(path: str, class_name: str) -> set[str]:
    import ast

    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                n.name
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("__")
            }
    raise AssertionError(f"{class_name} not found in {path}")


def test_extension_has_no_attr_conflict_with_npu_worker():
    """静态模拟 vllm worker_base.py:271 的冲突扫描(dir 扩展类的非 dunder 属性
    不得与 worker 类已有属性冲突)。NPUWorker 在纯 CPU 测试进程里不可导入
    (运行态依赖),故用 AST 取两边的方法集合 —— 比 import 更稳,且语义等价。
    """
    import vime.backends.megatron_utils.update_weight.update_weight_from_tensor as uwft

    ext_attrs = {
        n
        for n in _class_method_names(uwft.__file__, "vLLMColocateWorkerExtension")
    }
    import vllm_ascend  # 只取路径,不导入 worker(运行态依赖)

    worker_path = os.path.join(os.path.dirname(vllm_ascend.__file__), "worker", "worker.py")
    worker_attrs = _class_method_names(worker_path, "NPUWorker")

    conflict = ext_attrs & worker_attrs
    assert not conflict, f"扩展类与 NPUWorker 属性冲突: {conflict} —— WorkerProc 启动会断言失败"
    assert "update_weights_chunk" in ext_attrs


# ─── 2. 共卡/远程切分 ────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "offsets,counts,actor_gpus,expected",
    [
        # 混合满配 16+12:14 台 TP2,actor 16 卡 → 8 共卡 + 6 远程
        (list(range(0, 28, 2)), [2] * 14, 16, 8),
        # 注意:切分按槽位范围 [0, actor_gpus) 判定 —— 共卡/专用边界必须正好落在
        # actor 卡数上(当前 8+6 布局满足);共卡段若不足 actor 卡数(如 14),边界
        # 内缩会把第一台专用引擎误判为共卡,该布局不被支持。
        # 纯 colocate:8 台全共卡
        (list(range(0, 16, 2)), [2] * 8, 16, 8),
        # 纯分离(pd):所有引擎都在 actor 范围外
        ([16, 18, 20, 22], [2] * 4, 16, 0),
        # TP4 引擎:4 台,actor 16 → 全共卡
        ([0, 4, 8, 12], [4] * 4, 16, 4),
    ],
)
def test_count_colocated_engines(offsets, counts, actor_gpus, expected):
    assert count_colocated_engines(offsets, counts, actor_gpus) == expected


# ─── 3. 引擎级 offload 过滤(共卡 sleep、专用常驻) ──────────────────────────────
def _fake_server_group(*, needs_offload, has_share, shared_num_gpus, n_engines=14, per_engine=2):
    from vime.ray.rollout import ServerGroup

    g = object.__new__(ServerGroup)
    g.needs_offload = needs_offload
    g.all_engines = [object() for _ in range(n_engines)]
    g.num_gpus_per_engine = per_engine
    g.gpu_offset = 0
    spec = types.SimpleNamespace(rollout_has_share=has_share, rollout_shared_num_gpus=shared_num_gpus)
    g.args = types.SimpleNamespace(
        resource_layout_spec=spec,
        actor_num_nodes=1,
        actor_num_gpus_per_node=16,
        num_gpus_per_node=16,
    )
    return g


def test_offload_indices_hybrid_share():
    g = _fake_server_group(needs_offload=True, has_share=True, shared_num_gpus=16)
    assert g._offload_engine_indices() == list(range(8))  # 仅 8 台共卡引擎 sleep


def test_offload_indices_share_partial():
    # 共卡段只有 14 卡(HCCL 排他变体)时,第 8 台(.64 专用)不得误判 sleep
    g = _fake_server_group(needs_offload=True, has_share=True, shared_num_gpus=14)
    assert g._offload_engine_indices() == list(range(7))


def test_offload_indices_non_share_passthrough():
    g = _fake_server_group(needs_offload=True, has_share=False, shared_num_gpus=0, n_engines=8)
    assert g._offload_engine_indices() == list(range(8))
    g2 = _fake_server_group(needs_offload=False, has_share=False, shared_num_gpus=0, n_engines=8)
    assert g2._offload_engine_indices() == []


# ─── 4. IPC 借用显存 × layerwise 缓冲的生命周期回归 ─────────────────────────────
class _ToyLayer(nn.Module):
    """带 weight_loader 语义的玩具层(走 vllm 默认 loader)。"""

    def __init__(self, n):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n, n), requires_grad=False)


class _ToyLayer1(nn.Module):
    """两个参数:一个随 chunk0 交付(被缓冲),一个随 chunk1 交付(凑齐触发处理)。"""

    def __init__(self, n):
        super().__init__()
        self.w_a = nn.Parameter(torch.zeros(n, n), requires_grad=False)
        self.w_b = nn.Parameter(torch.zeros(n, n), requires_grad=False)


class _ToyModel(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layer0 = _ToyLayer(n)
        self.layer1 = _ToyLayer1(n)

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        for name, w in weights:
            param = params[name]
            param.weight_loader(param, w)


def _run_chunked_ipc_reload(clone_on_receive: bool, garbage_between_chunks: bool):
    """模拟:layer1 的 w_a 随 chunk0 交付但被缓冲(层未齐),trainer 随即覆写源显存,
    chunk1 交付 w_b 凑齐该层触发处理 —— 验证缓冲的 w_a 是否被污染。

    通过真实 layerwise reload 机制(record/initialize + online_process_loader 缓冲),
    并用假的 IPC rebuild 函数交付**借用视图**,复现跨进程生命周期风险。
    """
    from vllm.model_executor.model_loader.reload import layerwise as lw

    from vime.backends.megatron_utils.update_weight import update_weight_from_tensor as uwft

    n = 64
    model = _ToyModel(n)
    s0, sa, sb = torch.randn(n, n), torch.randn(n, n), torch.randn(n, n)
    s0_ref, sa_ref, sb_ref = s0.clone(), sa.clone(), sb.clone()  # pristine 对照

    lw.record_metadata_for_reloading(model)
    lw.initialize_layerwise_reload(model)

    # 假 IPC:func(*args) 返回源 tensor 的借用视图(uuid 与设备探测打桩)
    def make_fake_handle(src):
        def fake_rebuild(*args):
            return src.view(n, n)  # 借用视图,不持有新存储

        return {"fake-uuid": (fake_rebuild, (0,) * 7)}

    fake_dev = mock.Mock()
    fake_dev.current_device.return_value = 0
    fake_dev.get_device_properties.return_value = types.SimpleNamespace(uuid="fake-uuid")

    engine = types.SimpleNamespace(
        _weight_update_active=True,
        _is_checkpoint_format=True,
        model_runner=types.SimpleNamespace(model=model),
        device=torch.device("cpu"),
        vllm_config=None,
    )

    with (
        mock.patch.object(uwft, "_device_module", lambda: fake_dev),
        mock.patch("torch.accelerator.synchronize", lambda: None),
        mock.patch("vllm.config.set_current_vllm_config", lambda cfg: nullcontext()),
    ):
        if not clone_on_receive:
            # 反事实对照:取消 clone,验证测试确实能抓到借用显存问题
            orig = uwft.vLLMColocateWorkerExtension.update_weights_chunk

            def no_clone_chunk(self, update_info):
                with mock.patch("torch.Tensor.clone", lambda t: t):
                    return orig(self, update_info)

            chunk_fn = no_clone_chunk
        else:
            chunk_fn = uwft.vLLMColocateWorkerExtension.update_weights_chunk

        # chunk 0:layer0(完整,同步消费)+ layer1.w_a(缓冲,等 w_b)
        chunk0 = {
            "names": ["layer0.weight", "layer1.w_a"],
            "dtype_names": ["float32", "float32"],
            "shapes": [[n, n], [n, n]],
            "ipc_handles": [make_fake_handle(s0), make_fake_handle(sa)],
        }
        chunk_fn(engine, chunk0)

        if garbage_between_chunks:
            s0.fill_(float("nan"))  # trainer 侧 del + 复用:覆写源显存
            sa.fill_(float("nan"))

        # chunk 1:layer1.w_b —— 凑齐 layer1,触发处理(读缓冲的 w_a)
        chunk1 = {
            "names": ["layer1.w_b"],
            "dtype_names": ["float32"],
            "shapes": [[n, n]],
            "ipc_handles": [make_fake_handle(sb)],
        }
        chunk_fn(engine, chunk1)

    return model, (s0_ref, sa_ref, sb_ref)


def test_ipc_borrowed_tensor_lifetime_with_clone():
    model, (s0_ref, sa_ref, sb_ref) = _run_chunked_ipc_reload(clone_on_receive=True, garbage_between_chunks=True)
    assert torch.equal(model.layer0.weight, s0_ref), "layer0 被源显存覆写污染"
    assert torch.equal(model.layer1.w_a, sa_ref), "layer1.w_a 跨 chunk 缓冲期间被覆写污染"
    assert torch.equal(model.layer1.w_b, sb_ref), "layer1.w_b 被污染"


# ─── 5. 共享专家校验的就绪门控(run 172029 .64 失败根因) ───────────────────────
def _ready_gate(se_module):
    """以最小 fake 调用 vllm-ascend 的就绪门控(不构造 AscendFusedMoE)。"""
    from vllm_ascend.ops.fused_moe.fused_moe_0_23_0 import AscendFusedMoE

    fake_self = types.SimpleNamespace(_shared_experts=se_module)
    return AscendFusedMoE._shared_experts_ready_for_validation(fake_self)


def _make_info(load_numel, load_numel_total):
    from vllm.model_executor.model_loader.reload.types import LayerReloadingInfo

    info = LayerReloadingInfo(restore_metadata=({}, {}), restore_device=torch.device("cpu"))
    info.load_numel = load_numel
    info.load_numel_total = load_numel_total
    return info


def test_shared_expert_validation_gate():
    from vllm.model_executor.model_loader.reload.layerwise import LAYERWISE_INFO

    se = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    try:
        # 启动期(无会话条目)→ 就绪,保持启动校验
        assert _ready_gate(se) is True

        # reload 会话中,共享专家参数未到齐 → 未就绪(跳过,finish 兜底)
        LAYERWISE_INFO[se[0]] = _make_info(0, 16)
        assert _ready_gate(se) is False

        # 到齐 → 就绪
        LAYERWISE_INFO[se[0]] = _make_info(16, 16)
        assert _ready_gate(se) is True

        # 另一个子模块未初始化会话(total=None)→ 不阻塞
        LAYERWISE_INFO[se[1]] = _make_info(0, None)
        assert _ready_gate(se) is True
    finally:
        for m in (se[0], se[1]):
            LAYERWISE_INFO.pop(m, None)


# ─── 6. TCPStore 端口竞态修复(20260824 连续两次 EADDRINUSE)────────────────────
def test_dist_init_addr_parse():
    from vime.backends.vllm_utils.vllm_engine import parse_dist_init_addr

    assert parse_dist_init_addr("80.48.5.56:45000") == ("80.48.5.56", 45000)
    assert parse_dist_init_addr("[::1]:46001") == ("::1", 46001)


def test_dist_init_port_env_injection():
    """build_vllm_subprocess_env 把 dist_init_port 注入 VLLM_DIST_INIT_PORT。"""
    from vime.backends.vllm_utils.vllm_engine import build_vllm_subprocess_env

    env = build_vllm_subprocess_env(
        {
            "args": types.SimpleNamespace(vllm_enable_deterministic_inference=False, colocate=False),
            "dist_init_port": 45000,
            "visible_devices": "0,1",
        }
    )
    assert env["VLLM_DIST_INIT_PORT"] == "45000"


def test_vllm_executor_honors_dist_init_port_env():
    """vllm-023 multiproc_executor 必须读 VLLM_DIST_INIT_PORT 并回退 get_open_port。"""
    src = open("/workspace/vllm-023/vllm/v1/executor/multiproc_executor.py").read()
    assert 'os.environ.get("VLLM_DIST_INIT_PORT"' in src
    assert "or get_open_port()" in src


def test_cursor_allocation_unique_ports_per_node():
    """真实分配器 + 假引擎/假 ray.get:8(.56)+6(.64) 台引擎的 dist_init 端口同节点互斥。"""
    import ray

    from vime.ray.rollout import _allocate_rollout_engine_addr_and_ports_normal

    class FakeEngine:
        def __init__(self, node_idx):
            ip = "80.48.5.56" if node_idx == 0 else "80.48.5.64"
            self._get_current_node_ip_and_free_port = types.SimpleNamespace(
                remote=lambda start_port=None, consecutive=1: (ip, start_port)
            )

    engines = [(r, FakeEngine(0)) for r in range(8)] + [(r, FakeEngine(1)) for r in range(8, 14)]
    args = types.SimpleNamespace(
        rollout_num_gpus_per_engine=2,
        num_gpus_per_node=16,
        vllm_data_parallel_external_lb=False,
        vllm_dp_size=1,
    )

    with mock.patch.object(ray, "get", side_effect=lambda v: v):
        addr_and_ports, _ = _allocate_rollout_engine_addr_and_ports_normal(
            args=args,
            rollout_engines=engines,
            worker_type="regular",
            num_gpus_per_engine=2,
            rank_offset=0,
            base_port=15000,
        )

    ports_by_node: dict[str, list[int]] = {"80.48.5.56": [], "80.48.5.64": []}
    for rank, info in addr_and_ports.items():
        if rank > 13:
            continue  # 分配器会对节点补满 8 槽,14/15 是虚位,不验
        ip, port = info["dist_init_addr"].rsplit(":", 1)
        ports_by_node[ip].append(int(port))
    for ip, ports in ports_by_node.items():
        assert len(ports) == len(set(ports)), f"节点 {ip} 的 dist_init 端口重复: {sorted(ports)}"
    assert len(ports_by_node["80.48.5.56"]) == 8 and len(ports_by_node["80.48.5.64"]) == 6


def test_ipc_borrowed_tensor_without_clone_is_corrupted():
    # 反事实:没 clone 时,缓冲的 w_a 必然读到覆写后的 NaN(证明测试有效)
    model, (_s0_ref, sa_ref, _sb_ref) = _run_chunked_ipc_reload(clone_on_receive=False, garbage_between_chunks=True)
    assert torch.isnan(model.layer1.w_a).any() or not torch.equal(model.layer1.w_a, sa_ref)
