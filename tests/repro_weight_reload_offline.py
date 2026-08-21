"""离线全链路权重更新模拟 —— 不起集群,2 张 NPU 复现/验证引擎侧 reload 正确性。

背景:混合首跑 wv=1 时 .64 引擎(HCCL 路径,从不 sleep)在 reload 收尾的
FusedMoE 共享专家一致性校验失败。本脚本用**引擎自己的 checkpoint 做恒等
reload**——恒等载荷下校验必须过;过不了 = 引擎侧 reload 机制自身的 bug,
与 trainer/传输无关,可离线迭代修复。

场景:
  A  恒等 reload(等价 .64 引擎侧处理:start/finish 会话 + layerwise 处理 + 校验)
  B  sleep(1) → wake_up → reload(等价 .56 共卡引擎:wake 门控 + copy-back)

用法(引擎与训练同栈:走 editable finder,不要显式列 vllm 目录到 PYTHONPATH):
  ASCEND_RT_VISIBLE_DEVICES=8,9 python3 tests/repro_weight_reload_offline.py A
  ASCEND_RT_VISIBLE_DEVICES=8,9 python3 tests/repro_weight_reload_offline.py B
"""

import os
import sys
import time

MODEL = "/home/docker/Qwen3.6-35B-A3B-agentical-ascendc-hf-4t-bf16"


def main() -> None:
    scenario = sys.argv[1] if len(sys.argv) > 1 else "A"
    from vllm import LLM

    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        hf_overrides={"architectures": ["Qwen3_5MoeForConditionalGeneration"]},
        tensor_parallel_size=2,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enable_sleep_mode=True,
        enforce_eager=True,  # reload 校验与图无关,省 capture 时间
        additional_config={"multistream_overlap_shared_expert": True},
    )
    print("[repro] engine up", flush=True)

    if scenario in ("B", "AB"):
        print("[repro] sleep(level=1) ...", flush=True)
        llm.sleep(level=1)
        print("[repro] wake_up(tags=['weights']) ...", flush=True)
        llm.wake_up(tags=["weights"])
        time.sleep(2)

    print("[repro] identity reload_weights(is_checkpoint_format=True) ...", flush=True)
    t0 = time.time()
    llm.collective_rpc(
        "reload_weights",
        kwargs={"weights_path": MODEL, "is_checkpoint_format": True},
    )
    print(f"[repro] PASS —— 恒等 reload 完成且共享专家校验通过,耗时 {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
