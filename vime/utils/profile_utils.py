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
    def __init__(self, args):
        self.args = args
        self._torch_profiler_overall = None
        self._memory_profiler_overall = None
        self._stage_profilers = {}

        targets = list(args.profile_target) if args.use_pytorch_profiler else []
        # [2026-08-10 NPU] torch_npu 同一进程只允许一个活跃 profiler;
        # 多 target 在 NPU 上会撞 "profiler already running" → 只取第一个并告警。
        if is_npu() and len(targets) > 1:
            logger.warning(
                "[profiler] NPU 只允许单个活跃 profiler,targets=%s 只生效第一个(%s)。"
                "要采多个阶段请分多次跑。", targets, targets[0],
            )
            targets = targets[:1]

        if args.use_pytorch_profiler and ("train_overall" in targets):
            self._torch_profiler_overall = _create_torch_profiler(args, name="train_overall")
        for name in ("train_actor", "train_log_probs"):
            if args.use_pytorch_profiler and name in targets:
                self._stage_profilers[name] = _create_torch_profiler(args, name=name)

        if args.record_memory_history and ("train_overall" in targets):
            self._memory_profiler_overall = _BaseMemoryProfiler.create(args)
            self._memory_profiler_overall.start()

    def on_init_end(self):
        if self._torch_profiler_overall is not None:
            self._torch_profiler_overall.start()
        for p in self._stage_profilers.values():
            p.start()

    def step(self, rollout_id: int):
        if self._torch_profiler_overall is not None:
            self._torch_profiler_overall.step()

        if (
            self._memory_profiler_overall is not None
            and ((s := self.args.memory_snapshot_num_steps) is not None)
            and (rollout_id == s - 1)
        ):
            self._memory_profiler_overall.stop()

    @contextmanager
    def stage(self, name: str):
        """包一个训练阶段(train_actor=actor 前反向更新 / train_log_probs=log_prob 前向)。
        schedule 由 --profile-step-start/end 驱动:每次阶段出现 step 一次。"""
        p = self._stage_profilers.get(name)
        if p is None:
            yield
            return
        try:
            yield
        finally:
            p.step()

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


def _create_torch_profiler(args, name):
    if is_npu():
        # [2026-08-10] NPU 版:走 torch_npu.profiler,才能拿到 NPU 设备侧算子数据。
        # 参考 verl 昇腾采集指南:Level1 + 离线解析(analyse_flag=False,事后用
        # torch_npu.profiler.profiler.analyse 或 MindStudio Insight 打开)。
        import torch_npu

        trace_dir = (
            getattr(args, "tensorboard_dir", None)
            or os.environ.get("TENSORBOARD_DIR")
            or "outputs/profile"
        )
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        )
        return torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=torch_npu.profiler.schedule(
                wait=max(args.profile_step_start - 1, 0),
                warmup=1 if args.profile_step_start > 0 else 0,
                active=args.profile_step_end - args.profile_step_start,
                repeat=1,
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                trace_dir,
                worker_name=f"{name}_rank_{torch.distributed.get_rank()}",
                analyse_flag=False,   # 离线解析,避免在线解析拖垮训练
            ),
            record_shapes=True,
            with_modules=True,        # 框架层调用栈,膨胀低于 with_stack
            with_stack=False,
            profile_memory=False,
            experimental_config=experimental_config,
        )
    return torch.profiler.profile(
        schedule=torch.profiler.schedule(
            # TODO the train_actor and train_log_probs ones may need to have different args to control step
            wait=max(args.profile_step_start - 1, 0),
            warmup=1 if args.profile_step_start > 0 else 0,
            active=args.profile_step_end - args.profile_step_start,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            args.tensorboard_dir,
            worker_name=f"{name}_rank_{torch.distributed.get_rank()}",
            use_gzip=True,
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
