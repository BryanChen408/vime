#!/bin/bash
# ═══ Qwen3.5-4B(VLM + hybrid GatedDeltaNet,SWE 真实目标)一键启动 ═══
#   复用 start_swegym_8b.sh 的全部规模/对齐7项/CoT/LB-proxy fix,只 export 4B 的架构差异 env。
#   run-swegym-codex-polar.sh 已把 model-script / tool-parser / GDN-VLM args 做成 env-gated;
#   GDN/VLM 那套照抄用户已验证能跑的 run-qwen36-35b-polar-ppo.sh(35B MoE+GDN),4B 用 dense 版。
#
#   前置:① Qwen3.5-4B HF 权重 /home/docker/Qwen3.5-4B(modelscope download Qwen/Qwen3.5-4B)
#        ② torch_dist:bash scripts/swe/convert_qwen35_4b.sh → /home/docker/Qwen3.5-4B_torch_dist
#        ③ 磁盘够(当前满,须先清)。
#
#   换 4B 相对 8B 的 6 处差异(全在下面 export;run 脚本消费):
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# ── ③ 架构:MODEL_ARGS 换 qwen3_5 spec + GDN(替 qwen3-8B.sh dense)──
export MODEL_ARGS_SCRIPT=qwen3.5-4B.sh
# ── ① 权重路径 ──
export HF_CKPT=/home/docker/Qwen3.5-4B
# export REF_LOAD=/home/docker/Qwen3.5-4B_torch_dist
export REF_LOAD=/home/docker/Qwen3.5-4B_torch_dist_new
export SAVE=${SAVE:-/workspace/Qwen3.5-4B_vime_swegym}
# ── ② tool-parser:4B 发 <function=> XML(治 8B-instruct 的 hermes JSON 转义报错)──
export VLLM_TOOL_CALL_PARSER=qwen3_coder
# ── ④⑤⑥ GDN backend + VLM model-name/hf-overrides(FEAT_GDN 开;dense 版,非 35B moe)──
export FEAT_GDN=1
export VLLM_MODEL_NAME=qwen3_5forconditionalgeneration
export VLLM_HF_OVERRIDES='{"architectures":["Qwen3_5ForConditionalGeneration"]}'

# ── context:回归【官方上游 run.sh 默认】engine=50000(缓冲垫设计,非同事 override 的 32000)──
#   官方 run.sh 默认 SGLANG_CONTEXT_LENGTH=50000;同事在 launch_e2e 把它 override 降到 32000(省 GPU 显存)。
#   官方缓冲垫:engine 50000 ≥ prompt 累积(~32000)+ response 16000 → 多轮 agentic prompt 撞不到硬限,
#   消除 32001 reject/截断;NPU 显存够开 50000(用户确认)。
#   vime 侧 engine 硬限由 --vllm-max-model-len 决定(rollout-max-context-len 为 fallback),两个一起提到 50000。
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-50000}
export ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-50000}

# ── 迭代:num-rollout 200(用户指定;同事 num-epoch 20≈320)──
export NUM_ROLLOUT=${NUM_ROLLOUT:-200}

# ── 标识 ──
export RUN_ID=${RUN_ID:-swegym_codex_4b_$(date +%Y%m%d-%H%M%S)}
export WANDB_PROJECT=${WANDB_PROJECT:-polar-swegym-4b}

# batch(rollout-batch 4×n-samples 16)、async 2、min-complete 0.6、normalize-adv、eps-clip-high 0.28、
#   log-probs-chunk-size 256、CoT(polar 侧 POLAR_MASK_REASONING=0)、LB proxy fix —— 全部继承 start_swegym_8b.sh。
# 卡位(rollout 8/actor 4 TP=4)也先复用 8B;4B 更小,若 TP4 偏大可后续降 TP(先跑通)。
exec bash "${SCRIPT_DIR}/start_swegym_8b.sh"

# ── 看什么(4B 冒烟判据,相对 8B 的变化)────────────────────────────────
#   ✅ Risk#1:completions≠0 + hermes 报错应≈0(qwen3_coder XML 不需 JSON 转义 → 治好 8B 的 tool-call 报错)
#   ✅ 32001 超长应≈0(4B native 256k,40960 不撞 edge)、truncated 应<<8B 的 22%
#   ✅ reward 应比 8B(卡 ~1.5%)高:4B 是同事真实目标 + coding tool-call 更规范 → 组内更易 resolved
#   ⚠️ GDN 首启注意:--qwen-gdn-backend npu 的 MindSpeed 算子(chunk_gated_delta_rule/causal_conv1d)加载不报错;
#      vllm 加载 Qwen3_5ForConditionalGeneration(VLM,只用文本)不崩(model-name/hf-overrides 正确)。
