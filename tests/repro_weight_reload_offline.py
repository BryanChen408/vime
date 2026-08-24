"""离线全链路权重更新模拟 —— 不起集群,2 张 NPU 复现/验证引擎侧 reload 正确性。

背景:混合首跑 wv=1 时 .64 引擎(HCCL 路径,从不 sleep)在 reload 收尾的
FusedMoE 共享专家一致性校验失败。本脚本用**引擎自己的 checkpoint 做恒等
reload**——恒等载荷下校验必须过;过不了 = 引擎侧 reload 机制自身的 bug,
与 trainer/传输无关,可离线迭代修复。

场景:
  A  恒等 reload(等价 .64 引擎侧处理:start/finish 会话 + layerwise 处理 + 校验)
  B  sleep(1) → wake_up → reload(等价 .56 共卡引擎:wake 门控 + copy-back)
  C  微扰 reload(共享专家 shard ×1.5,模拟一步训练漂移)—— A 过而 C 挂
     ⟹ reload 机制对"变化的权重"处理有 bug(陈旧拷贝/布局),可离线二分定位。
  D  生产保真投喂:start/finish 会话 + 按 **direct 迭代器的字母序** 分块投喂
     (checkpoint 原值)—— D 挂 ⟹ 投喂顺序/层完成时序就是 .64 失败根因;
     D 过 ⟹ 只剩 trainer 转换值本身,下一步对账 trainer 输出与 checkpoint。

  追加维度(与 A/B/C/D 组合,如 AE、BE):
  场景字母后带 E 表示**关闭 enforce_eager**(用生产同款 FULL_DECODE_ONLY 图),
  用于检验图捕获/多流状态对校验的影响(离线默认 enforce_eager=True)。

用法(引擎与训练同栈:走 editable finder,不要显式列 vllm 目录到 PYTHONPATH):
  ASCEND_RT_VISIBLE_DEVICES=8,9 python3 tests/repro_weight_reload_offline.py A
"""

import os
import sys
import time

MODEL = "/home/docker/Qwen3.6-35B-A3B-agentical-ascendc-hf-4t-bf16"


def run_scenario_d(llm) -> None:
    """生产保真投喂:start 会话 → 按 direct 迭代器的字母序分块喂 checkpoint 原值 → finish。

    direct 迭代器(hf_weight_iterator_direct.py:190)按名字**字母序**排序参数
    (layers.0, .1, .10, .11, ...),与 PD 的"非专家先行、专家第二轮"顺序不同。
    若 D 挂 → 投喂顺序导致的层完成时序就是 .64 失败根因。
    """
    import json

    idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))
    weight_map = idx["weight_map"]
    # 引擎未启用 MTP(spec decode 关闭),checkpoint 里的 mtp.* 权重引擎模型没有,跳过
    names = sorted(n for n in weight_map if not n.startswith("mtp."))  # 字母序,模拟 direct 迭代器
    # 按 ~4GB 分桶(与 update_weight_buffer_size 量级一致)
    bucket, buckets, size = [], [], 0
    for name in names:
        shard = os.path.join(MODEL, weight_map[name])
        # 粗估 bf16 2 字节/元素
        from safetensors import safe_open

        with safe_open(shard, framework="pt") as f:
            numel = 1
            for d in f.get_slice(name).get_shape():
                numel *= d
        nbytes = numel * 2
        if bucket and size + nbytes > 4 * 1024**3:
            buckets.append(bucket)
            bucket, size = [], 0
        bucket.append((name, shard))
        size += nbytes
    if bucket:
        buckets.append(bucket)
    print(f"[repro-D] {len(names)} 参数,{len(buckets)} 桶,字母序投喂", flush=True)

    t0 = time.time()
    llm.collective_rpc("start_weight_update", kwargs={"is_checkpoint_format": True})
    for i, bucket in enumerate(buckets):
        llm.collective_rpc(
            "debug_feed_chunk",
            kwargs={"update_info": {"names": [n for n, _ in bucket], "shards": [s for _, s in bucket]}},
        )
        print(f"[repro-D] bucket {i + 1}/{len(buckets)} fed ({time.time() - t0:.0f}s)", flush=True)
    llm.collective_rpc("finish_weight_update")
    print(f"[repro-D] PASS —— 字母序投喂完成且校验通过,耗时 {time.time() - t0:.1f}s", flush=True)


def main() -> None:
    scenario = sys.argv[1] if len(sys.argv) > 1 else "A"
    eager = True
    if scenario.endswith("E"):
        eager = False  # 生产同款 FULL_DECODE_ONLY 图;校验图/多流状态的影响
        scenario = scenario[:-1]
    do_generate = scenario.endswith("G")
    if do_generate:
        scenario = scenario[:-1]  # 先跑一遍 generate(模拟 wv=1 前的 rollout 窗口)
    from vllm import LLM

    is_d = scenario == "D"
    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        hf_overrides={"architectures": ["Qwen3_5MoeForConditionalGeneration"]},
        tensor_parallel_size=2,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enable_sleep_mode=True,
        enforce_eager=eager,
        additional_config={"multistream_overlap_shared_expert": True},
        **(
            {
                "worker_extension_cls": "repro_ext.DebugReloadExtension",
                "weight_transfer_config": {"backend": "npu_ipc"},
            }
            if is_d
            else {}
        ),
    )
    print("[repro] engine up", flush=True)

    if do_generate:
        from vllm import SamplingParams

        print("[repro] warmup generate (模拟 rollout 窗口) ...", flush=True)
        outs = llm.generate(["你好,请用一句话介绍你自己。"] * 4, SamplingParams(max_tokens=32, temperature=0.8))
        print(f"[repro] generate done, sample: {outs[0].outputs[0].text[:40]!r}", flush=True)

    if scenario in ("B", "AB"):
        print("[repro] sleep(level=1) ...", flush=True)
        llm.sleep(level=1)
        print("[repro] wake_up(tags=['weights']) ...", flush=True)
        llm.wake_up(tags=["weights"])
        time.sleep(2)
    elif scenario == "B2":
        # level=2 丢弃式 sleep(slime colo 语义):权重不 park host,wake 重映射空壳,
        # reload 全量覆写 —— 验证"丢弃 + wake + reload 会话"组合的正确性
        print("[repro] sleep(level=2,丢弃不驻留) ...", flush=True)
        llm.sleep(level=2)
        print("[repro] wake_up(tags=['weights'],空壳重映射) ...", flush=True)
        llm.wake_up(tags=["weights"])
        time.sleep(2)

    if is_d:
        run_scenario_d(llm)
        return

    print("[repro] identity reload_weights(is_checkpoint_format=True) ...", flush=True)
    t0 = time.time()
    if scenario == "C":
        # 微扰:软链全部文件,仅重写含共享专家权重的 shard 7(全部 ×1.5,
        # 模拟"变化了的权重"),其余 shard 恒等。A 过而 C 挂 ⟹ reload 机制对
        # 变化权重的处理有 bug(陈旧拷贝/布局),可离线二分。
        perturbed_dir = "/tmp/ckpt_perturbed"
        target_shard = "model-00007-of-00026.safetensors"
        marker = os.path.join(perturbed_dir, ".done")
        if not os.path.exists(marker):
            os.makedirs(perturbed_dir, exist_ok=True)
            for f in os.listdir(MODEL):
                dst = os.path.join(perturbed_dir, f)
                if not os.path.exists(dst):
                    os.symlink(os.path.join(MODEL, f), dst)
            from safetensors.torch import load_file, save_file

            print(f"[repro] building perturbed shard {target_shard} ...", flush=True)
            tensors = load_file(os.path.join(MODEL, target_shard))
            tensors = {k: v * 1.5 for k, v in tensors.items()}
            dst = os.path.join(perturbed_dir, target_shard)
            if os.path.islink(dst):
                os.unlink(dst)
            save_file(tensors, dst, metadata={"format": "pt"})
            open(marker, "w").write("ok")
        llm.collective_rpc(
            "reload_weights",
            kwargs={"weights_path": perturbed_dir, "is_checkpoint_format": True},
        )
    else:
        llm.collective_rpc(
            "reload_weights",
            kwargs={"weights_path": MODEL, "is_checkpoint_format": True},
        )
    print(f"[repro] PASS —— reload 完成且共享专家校验通过,耗时 {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
