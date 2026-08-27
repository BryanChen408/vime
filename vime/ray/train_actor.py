import abc
import logging
import os
import random
from datetime import timedelta

import ray
import torch
import torch.distributed as dist

import vime.utils.eval_config
from vime.ray.ray_actor import RayActor
from vime.utils.distributed_utils import init_gloo_group
from vime.utils.logging_utils import configure_logger
from vime.utils.memory_utils import (
    aggressive_empty_cache,
    clear_memory,
    expandable_segments_enabled,
    print_memory,
    set_expandable_segments,
)
from vime.utils.common import is_npu

logger = logging.getLogger(__name__)


def get_local_gpu_id():
    if is_npu():
        env_var = "ASCEND_RT_VISIBLE_DEVICES"
        device_ids = ray.get_runtime_context().get_accelerator_ids()["NPU"]
    else:
        env_var = "CUDA_VISIBLE_DEVICES"
        device_ids = ray.get_gpu_ids()
    cvd = os.environ.get(env_var, None)
    if cvd is None:
        return device_ids[0]
    else:
        return cvd.split(",").index(str(device_ids[0]))

class TrainRayActor(RayActor):
    def __init__(self, world_size, rank, master_addr, master_port):
        configure_logger()

        self._world_size = world_size
        self._rank = rank
        if master_addr:
            self.master_addr, self.master_port = master_addr, master_port
        else:
            self.master_addr, self.master_port = self._get_current_node_ip_and_free_port(
                start_port=random.randint(20000, 21000)
            )

        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # TODO: currently this doesn't work as ray has already set torch.cuda.device_count().
        # os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        # os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0])
        os.environ["LOCAL_RANK"] = str(get_local_gpu_id())

    def init(self, args, role, with_ref=False, with_opd_teacher=False):
        self.args = args
        self.role = role
        self.with_ref = with_ref
        self.with_opd_teacher = with_opd_teacher
        self._memory_handoff_active = False
        self._restore_expandable_segments = False

        torch.serialization.add_safe_globals([vime.utils.eval_config.EvalDatasetConfig])

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if is_npu():
            torch.npu.set_device(f"npu:{local_rank}")
        else:
            torch.cuda.set_device(f"cuda:{local_rank}")

        backend = args.distributed_backend

        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
        )
        init_gloo_group()

        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()

        try:
            import pynvml

            pynvml.nvmlInit()

            local_rank = int(os.environ["RANK"]) % args.num_gpus_per_node

            handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
            pynvml.nvmlDeviceSetCpuAffinity(handle)

            logger.info(f"Set NUMA affinity for GPU {local_rank}")
            pynvml.nvmlShutdown()

        except ImportError:
            logger.info("Warning: pynvml not available, skipping NUMA affinity setup")
        except Exception as e:
            logger.info(f"Warning: Failed to set NUMA affinity: {e}")

    def clear_memory(self):
        if self.args.debug_rollout_only:
            return
        print_memory("before TrainRayActor.clear_memory")
        clear_memory()
        print_memory("after TrainRayActor.clear_memory")

    def prepare_memory_handoff(self) -> None:
        """Prepare a colocated NPU actor allocator before rollout weights wake."""
        if not (
            is_npu()
            and getattr(self.args, "offload_train", False)
            and getattr(self, "_rollout_shares_actor_devices", False)
        ):
            return
        if self._memory_handoff_active:
            logger.info("NPU memory handoff is already active; skipping duplicate prepare")
            return

        self._memory_handoff_active = True
        self._restore_expandable_segments = expandable_segments_enabled()

        # Port the allocator bracket used by verl: synchronization/update
        # temporaries must be allocated while expandable segments are disabled,
        # so they can be returned cleanly before vLLM remaps its KV cache.
        set_expandable_segments(False)
        aggressive_empty_cache(force_sync=True)
        print_memory("after memory handoff prepare (before rollout weights wake)")

    def finish_memory_handoff(self) -> None:
        """Close a successful handoff after rollout KV cache is fully awake."""
        if not getattr(self, "_memory_handoff_active", False):
            return

        try:
            # mem_get_info is device-global, so this records the trainer residue
            # together with the fully restored vLLM weights/KV/graphs.
            print_memory("after rollout KV cache wake")
        finally:
            if self._restore_expandable_segments:
                set_expandable_segments(True)
            self._restore_expandable_segments = False
            self._memory_handoff_active = False

    @abc.abstractmethod
    def sleep(self, tags):
        raise NotImplementedError

    @abc.abstractmethod
    def wake_up(self, tags):
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, rollout_id, rollout_data_ref, external_data=None):
        raise NotImplementedError

    @abc.abstractmethod
    def save_model(self, rollout_id, force_sync=False):
        raise NotImplementedError

    @abc.abstractmethod
    def update_weights(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _get_parallel_config(self):
        raise NotImplementedError

    def set_rollout_manager(self, rollout_manager):
        self.rollout_manager = rollout_manager
        if not self.args.debug_rollout_only and self.args.rank == 0:
            ray.get(self.rollout_manager.set_train_parallel_config.remote(self.train_parallel_config))
