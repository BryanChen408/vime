"""受控实验:sleep(level=2)/wake 前后,解码速度是否退化(验证"生成爬成 0.5 tok/s"根因)。

对比:fresh 测速 → sleep(2) → wake → 再测速。若 wake 后掉到 ~0.5 tok/s,
即坐实 sleep/wake 弄废解码;若仍 ~50 tok/s,则排除 sleep/wake,0.5 tok/s 是长上下文工作负载。
"""
import os
import time

MODEL = "/home/docker/Qwen3.6-35B-A3B-agentical-ascendc-hf-4t-bf16"


def measure(llm, tag, n=30):
    from vllm import SamplingParams

    t0 = time.time()
    outs = llm.generate(["写一段50字自我介绍"] * 4, SamplingParams(max_tokens=n, temperature=0))
    dt = time.time() - t0
    toks = sum(len(o.outputs[0].token_ids) for o in outs)
    print(f"[{tag}] {toks} tok / {dt:.1f}s = {toks/dt:.1f} tok/s", flush=True)


def main():
    from vllm import LLM

    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        hf_overrides={"architectures": ["Qwen3_5MoeForConditionalGeneration"]},
        tensor_parallel_size=2,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        enable_sleep_mode=True,
        enforce_eager=True,
        additional_config={"multistream_overlap_shared_expert": True},
    )
    print("[t] engine up", flush=True)
    measure(llm, "fresh")
    print("[t] sleep(level=2) ...", flush=True)
    llm.sleep(level=2)
    print("[t] wake_up ...", flush=True)
    llm.wake_up()
    measure(llm, "wake(level=2)后")


if __name__ == "__main__":
    main()
