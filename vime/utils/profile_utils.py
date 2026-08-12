import logging
import os
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import torch

from vime.utils.common import is_npu
from vime.utils.memory_utils import print_memory

logger = logging.getLogger(__name__)


class TrainProfiler:
    """训练侧 profiling(2026-08-12 重做,照 verl mstx_profile.py 的实证模式)。

    核心:不用 schedule。这版 torch_npu 的 schedule 语义反直觉(WARMUP 段也真实录制、
    RECORD_AND_SAVE 在步边界才落盘,连踩两坑)——改为"命中目标步才建 profiler:
    start → 跑 → step → stop",录制窗精确等于目标本身。
    阶段 profiler 用完即停,NPU 单活跃约束天然满足 → 多个 target 可在同一 run 都采。
    """

    def __init__(self, args):
        self.args = args
        self._enabled = bool(args.use_pytorch_profiler)
        self._targets = set(args.profile_target) if self._enabled else set()
        # 用户语义:PROFILE_STEP_START=N → 采第 N 个 train 步(1-based)。
        # rollout_id 是 0-based(第一轮 id=0)→ 命中区间 [START-1, END-1)。
        self._step_lo = args.profile_step_start - 1
        self._step_hi = args.profile_step_end - 1
        # rank 门(verl 同款:开启但未指定 PROFILE_RANKS → 只采 rank 0)
        self._this_rank = True
        if self._enabled and torch.distributed.is_available() and torch.distributed.is_initialized():
            _env = os.environ.get("PROFILE_RANKS")
            _allow = {int(x) for x in _env.split(",") if x.strip()} if _env else {0}
            self._this_rank = torch.distributed.get_rank() in _allow
        self._memory_profiler_overall = None
        if (
            self._enabled and self._this_rank
            and args.record_memory_history and ("train_overall" in self._targets)
        ):
            self._memory_profiler_overall = _BaseMemoryProfiler.create(args)
            self._memory_profiler_overall.start()

    def on_init_end(self):
        # 懒启动:不预建不预启 —— 预启会把启动/等待期也录进去(踩过)。
        pass

    def _want(self, name, rollout_id):
        if not (self._enabled and self._this_rank):
            return False
        if rollout_id is None or not (self._step_lo <= rollout_id < self._step_hi):
            return False
        if name in self._targets:
            return True
        # 兼容旧 target:train_log_probs = ref/teacher/actor 三个 log_prob 阶段的集合
        return "train_log_probs" in self._targets and name.endswith("_log_probs")

    @contextmanager
    def stage(self, name: str, rollout_id: int = None):
        """包一个阶段/整步。命中:进入时新建 profiler 并 start,退出时 step+stop 落盘。"""
        if not self._want(name, rollout_id):
            yield
            return
        prof = _create_torch_profiler(self.args, name=name, rollout_id=rollout_id)
        prof.start()
        try:
            yield
        finally:
            try:
                prof.step()
            finally:
                prof.stop()

    def step(self, rollout_id: int):
        if (
            self._memory_profiler_overall is not None
            and ((s := self.args.memory_snapshot_num_steps) is not None)
            and (rollout_id == s - 1)
        ):
            self._memory_profiler_overall.stop()

    def iterate_train_actor(self, iterator):
        return _profile_simple_loop(iterator, self.args, name="train_actor")

    def iterate_train_log_probs(self, iterator):
        return _profile_simple_loop(iterator, self.args, name="train_log_probs")


def _profile_simple_loop(iterator, args, name):
    if not (args.use_pytorch_profiler and (name in args.profile_target)):
        yield from iterator
        return

    torch_profiler = _create_torch_profiler(args, name=name)
    torch_profiler.start()
    for item in iterator:
        yield item
        torch_profiler.step()


def _create_torch_profiler(args, name, rollout_id=None):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    trace_dir = (
        getattr(args, "tensorboard_dir", None)
        or os.environ.get("TENSORBOARD_DIR")
        or "outputs/profile"
    )
    worker = f"{name}_step{rollout_id}_rank_{rank}" if rollout_id is not None else f"{name}_rank_{rank}"
    if is_npu():
        # [2026-08-12] NPU 版:无 schedule,由 stage() 的 start/step/stop 驱动。
        # experimental_config 照 verl mstx_profile.py:Db 导出 + 数据精简 + msprof_tx;
        # PROFILE_LEVEL=level0/1/2(默认 level1);PROFILE_EXCLUDE_COMM=1 可排通信域降噪。
        import torch_npu

        _levels = {
            "level0": torch_npu.profiler.ProfilerLevel.Level0,
            "level1": torch_npu.profiler.ProfilerLevel.Level1,
            "level2": torch_npu.profiler.ProfilerLevel.Level2,
        }
        _level = _levels.get(os.environ.get("PROFILE_LEVEL", "level1").lower(),
                             torch_npu.profiler.ProfilerLevel.Level1)
        _exp = dict(
            profiler_level=_level,
            export_type=torch_npu.profiler.ExportType.Db,
            data_simplification=True,
            msprof_tx=True,
        )
        if os.environ.get("PROFILE_EXCLUDE_COMM", "0") == "1":
            _exp["mstx_domain_exclude"] = ["communication"]
        # 采集内容开关(照 verl contents 列表风格):PROFILE_CONTENTS="shapes,module,memory,stack"
        # 不设 = 全关(最省体积);按需打开。stack 体积最大,慎用。
        _contents = {x.strip() for x in os.environ.get("PROFILE_CONTENTS", "").split(",") if x.strip()}
        return torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                trace_dir, worker_name=worker, analyse_flag=False,  # 离线解析
            ),
            record_shapes="shapes" in _contents,
            with_modules="module" in _contents,
            with_stack="stack" in _contents,
            profile_memory="memory" in _contents,
            experimental_config=torch_npu.profiler._ExperimentalConfig(**_exp),
        )
    return torch.profiler.profile(
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            trace_dir, worker_name=worker, use_gzip=True,
        ),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        with_flops=True,
    )


class _BaseMemoryProfiler:
    @staticmethod
    def create(args):
        c = {
            "torch": _TorchMemoryProfiler,
            "memray": _MemrayMemoryProfiler,
        }[args.memory_recorder]
        return c(args)

    def __init__(self, args):
        self._path_dump = (
            Path(args.memory_snapshot_dir)
            / f"memory_snapshot_time{time.time()}_rank{torch.distributed.get_rank()}_{args.memory_snapshot_path}"
        )

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError


class _TorchMemoryProfiler(_BaseMemoryProfiler):
    def start(self):
        if is_npu():
            # torch.cuda.memory._record_memory_history 是 CUDA-only;NPU 侧暂无等价物,
            # 先告警跳过(不影响算子级 profiling)。
            logger.warning("[profiler] record_memory_history 暂不支持 NPU,跳过内存快照采集。")
            return
        logger.info("Attach OOM dump memory history.")

        torch.cuda.memory._record_memory_history(
            max_entries=1000000,
            # record stack information for the trace events
            # trace_alloc_record_context=True,
            stacks="all",
        )

        def oom_observer(device, alloc, device_alloc, device_free):
            logger.info(
                f"Observe OOM, will dump snapshot to {self._path_dump}. ({device=} {alloc=} {device_alloc=} {device_free=}; stacktrace is as follows)"
            )
            traceback.print_stack()
            torch.cuda.memory._dump_snapshot(self._path_dump)
            print_memory("when oom")

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    def stop(self):
        if is_npu():
            return
        logger.info(f"Dump memory snapshot to: {self._path_dump}")
        torch.cuda.memory._dump_snapshot(self._path_dump)
        torch.cuda.memory._record_memory_history(enabled=None)


class _MemrayMemoryProfiler(_BaseMemoryProfiler):
    def __init__(self, args):
        super().__init__(args)
        assert args.memory_snapshot_num_steps is not None, "In memray, must provide --memory-snapshot-num-steps"

    def start(self):
        logger.info("Memray tracker started.")
        import memray

        self._tracker = memray.Tracker(
            file_name=self._path_dump,
            native_traces=True,
        )
        self._tracker.__enter__()

    def stop(self):
        logger.info(f"Memray tracker stopped and dump snapshot to: {self._path_dump}")
        self._tracker.__exit__(None, None, None)
