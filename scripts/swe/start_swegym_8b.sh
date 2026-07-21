#!/bin/bash
# ═══ Qwen3-8B(dense)SWE-Gym coding-agent(codex)polar-agentic RL 一键启动 ═══
#   与 start_math_8b.sh 同一 polar-agentic 链路 / 同一 dense-8B 基座,换域 math→SWE:
#     · 数据:swegym_train_64.jsonl(Phase 3 产物,已含 metadata.docker_image/instance)——离线,不拉 HF。
#     · 提交:task_request(--polar-task-template);SWE 每题 docker 镜像不同 → 按 sample 渲染 runtime.image。
#     · agent:codex(hermes tool-parser);evaluator:swebench_harness(patch→跑测→resolved→reward)。
#   · 推理端点 :8001 = LB proxy(保 return_token_ids)→ FEAT_LB_PROXY=1;polar 用 profile.swe-8b.yaml。
#   · 卡位:SWE agent CPU-only(codex 改文件 + 跑 pytest,不需 NPU-in-container)→ polar 不占 NPU(0-3 空);
#           8B 用 4-15:rollout 8 卡 = 4 引擎 TP=2(DP4,LB proxy 按 session 亲和,非 external-LB DP → 无假前向 2.3×),
#           actor 4 卡 TP=4。
#
# ── 前置 ────────────────────────────────────────────────────────────────
#   1) 权重:/home/docker/Qwen3-8B(HF)+ /home/docker/Qwen3-8B_torch_dist(convert-qwen3-8B.sh 产物)。
#      (SWE 最终目标模型 = Qwen3-4B;此处先用已就绪的 8B 打通管线,见 DEV_PLAN。)
#   2) codex CLI:${AGENT_CLI_DIR}/bin/{node,codex}(@openai/codex@0.144.5;见 Phase 6 provisioning)。
#   3) 64 SWE docker 镜像已 docker load 到 polar 宿主机(镜像名 = jsonl 里的 metadata.docker_image)。
#   4) polar 以 SWE profile 启动(推理指向 vime :8001、docker runtime、swebench 包已装):
#      POLAR_PROFILE=.../deploy/ascend_operator/profile.swe-8b.yaml bash .../restart_polar_host.sh
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# ── 数据:Phase 3 已生成的 64 题 SWE jsonl(离线;无 math prep 步骤)──
export SWEGYM_JSONL=${SWEGYM_JSONL:-/home/docker/datasets/swegym/swegym_train_64.jsonl}
if [ ! -f "${SWEGYM_JSONL}" ]; then
   echo "[start_swegym_8b][FATAL] 缺 ${SWEGYM_JSONL} —— 先跑 scripts/swe/prepare_swegym_data.py(离线)。" >&2
   exit 1
fi

# ── codex CLI 目录(run 脚本 sed 渲染进模板 volume,挂进任务容器 /opt/node:ro)──
export AGENT_CLI_DIR=${AGENT_CLI_DIR:-/home/docker/datasets/swe_agent_cli}

# ── 拓扑 / 卡位(单机,8B,12 卡 4-15;agentic rollout 是瓶颈 → 卡大头给推理)──
#   VIME_ROLLOUT_LOW_CARDS=1 → rollout 4-11 / actor 12-15。SWE agent CPU-only,故 0-3 空(math 也占前 4 给算子评测,SWE 不需)。
export MASTER_ADDR=${MASTER_ADDR:-80.48.5.88}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-4,5,6,7,8,9,10,11,12,13,14,15}
export VIME_ROLLOUT_LOW_CARDS=1
export NPUS_PER_NODE=${NPUS_PER_NODE:-12}
export ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-8}
export ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}   # 4 引擎 × TP=2
export ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-4}
export TP=${TP:-4}                                                      # 训练 TP=4(4 actor 卡)

# ── 冒烟规模(SWE 比 math 重:codex 多轮 + 容器跑测,故 batch / active-sessions 保守)──
#   SWE 域值取自同事 run.sh:rollout-max-response-len=16000。context/seq/max-tokens 用 8B-NPU 已验预算
#     (start_math_8b 同款 SEQ 40960 / MAX_TOKENS_PER_GPU 32768,不照抄同事 GPU 的 20000)。
#   n_samples=8 造 GRPO 组内方差(swebench resolved=0/1,组内需偶有解出才有 advantage 信号,见 Risk #3)。
#   POLAR_MAX_ASYNC_LEVEL=1(对齐 math-8b、更 on-policy 便于看涨;同事 SWE 用 2 → 设 2 追其 staleness/吞吐)。
RUN_ID=${RUN_ID:-swegym_codex_8b_$(date +%Y%m%d-%H%M%S)} \
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8} \
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8} \
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64} \
NUM_ROLLOUT=${NUM_ROLLOUT:-50} \
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-32768} \
SEQ_LENGTH=${SEQ_LENGTH:-40960} \
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-32} \
VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.85} \
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-40960} \
ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-40960} \
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-16000} \
POLAR_MAX_ACTIVE_SESSIONS=${POLAR_MAX_ACTIVE_SESSIONS:-16} \
POLAR_MAX_ASYNC_LEVEL=${POLAR_MAX_ASYNC_LEVEL:-1} \
VIME_MEM_PROBE=1 FEAT_TRAIN_EXPANDABLE=1 VIME_EMPTY_CACHE_PER_STEP=1 \
FEAT_PREFIX_CACHE=1 FEAT_STATIC_KERNEL=1 FEAT_HCCL_AIV=1 \
FEAT_LB_PROXY=1 \
FEAT_WANDB=${FEAT_WANDB:-1} \
WANDB_KEY=${WANDB_KEY:-} \
WANDB_PROJECT=${WANDB_PROJECT:-polar-swegym-8b} \
TRAIN_EXTRA_ARGS=${TRAIN_EXTRA_ARGS:-"--skip-eval-before-train --save-debug-rollout-data /home/docker/logs/dbgroll_swegym_{rollout_id}.pt"} \
bash "${SCRIPT_DIR}/run-swegym-codex-polar.sh"

# ── 看什么(判据,SWE 冒烟)────────────────────────────────────────────
#   ✅ Risk #1 闸:completions≠0 —— codex↔vllm-ascend(/responses)通 + hermes 解到 <tool_call>;
#              全 0 = tool turn 坏(先查 parser / responses wire-api,别急着调训练)。
#   ✅ 评测链路:swebench_harness 对 gold-patch 判 resolved=True(docker 后端 + 跑测通)。
#   ✅ token-faith:train/tis≈1;reward 组内有方差(至少偶有 resolved=1),再看 mean 趋势。
#   ⚠️ 组内全 0(8B 解不动 SWE)→ 换更强/coding 模型或挑简单子集(Risk #3 冷启动)。
#   注:FEAT_WANDB=1 但 WANDB_KEY 空 → run 脚本自动不开 wandb(需设本机 wandb key 才上报)。
