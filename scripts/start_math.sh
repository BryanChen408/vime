#!/bin/bash
# ═══ math agentic 管线验证一键启动(DAPO-17k → polar claudecode + math_judge)═══
#
# 目的:验证 polar 接入整条 vime RL 管线是否正确(轨迹捕获→rebuild→reward→advantage→回灌→
#   策略改善)。判据:reward 趋势上升 + train/tis≈1。不是训模型,是给管线冒烟。
#
# ── 隔离契约(关键,务必看懂再改)────────────────────────────────────────
#   本脚本【零改动】算子的 run-qwen36-35b-polar-minimal.sh 与整个 vime_bridge:
#     · math 与 operator 走【同一条】已验证的 operator_samples 派发/推理/轨迹捕获链路;
#     · 与 operator 的唯一差别 = 不 attach task_source(agent 只拿题面 instruction,不拿任务文件)。
#       实现:调用【姊妹脚本 run-qwen36-35b-math.sh】= 原版 minimal 删掉唯一那行 --operator-tasks-dir
#       (原版一字不动)。删该行 → args.operator_tasks_dir=None(argparse default=None 可选)→
#       _attach_operator_task_source 早返回 → 不传 task_source。
#     · 推理端点 :8001 = LB proxy(保 return_token_ids)→ 本脚本置 FEAT_LB_PROXY=1;profile.math 指 :8001。
#       (裸 Rust router :8001 会丢 return_token_ids → session 全 ERROR;单引擎直连则改 :15000 且去 FEAT_LB_PROXY。)
#     · 答案在 sample.metadata.answer(judge-only,不进 agent prompt);经 operator_profile.py:194
#       → task_metadata.sample_metadata.answer 到 math_judge。
#     · profile 由 polar 侧 default_operator_profile=math_npu 决定;vime 不传 profile,只发 URL。
#
# ── 前置(polar 侧,一次性)──────────────────────────────────────────────
#   polar 必须以 math profile 启动(math 代码/profile 在隔离分支 `math` @ /home/docker/math_wt):
#     POLAR_PROFILE=deploy/ascend_operator/profile.math.yaml bash restart_polar_host.sh
#
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# ── 数据:DAPO-17k → operator_samples 行(prompt 保持消息列表;prep 补 metadata.op_name + answer)──
RAW_DAPO=${RAW_DAPO:-/home/docker/datasets/dapo-math-17k.jsonl}
export OPERATOR_TASK_JSONL=${OPERATOR_TASK_JSONL:-/home/docker/datasets/dapo-math-17k-prep.jsonl}
if [ ! -f "${OPERATOR_TASK_JSONL}" ]; then
   echo "[start_math] prep DAPO(+metadata.op_name/answer,prompt 保持 list)→ ${OPERATOR_TASK_JSONL}"
   python3 "${SCRIPT_DIR}/prep_dapo_math.py" "${RAW_DAPO}" "${OPERATOR_TASK_JSONL}"
fi

# ── 拓扑 / 卡位(单机,rollout 4-7 / actor 8-15)──
export MASTER_ADDR=${MASTER_ADDR:-80.48.5.88}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-4,5,6,7,8,9,10,11,12,13,14,15}
export VIME_ROLLOUT_LOW_CARDS=1

# ── 冒烟规模:bs8 × N8 = 64 序列/step,造组内 reward 方差(避免 zero-std 假阴性)──
RUN_ID=${RUN_ID:-qwen36_math_$(date +%Y%m%d-%H%M%S)} \
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8} \
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8} \
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-64} \
NUM_ROLLOUT=${NUM_ROLLOUT:-50} \
QWEN36_CHUNK_LMHEAD=1 \
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-32768} \
POLAR_MAX_ACTIVE_SESSIONS=${POLAR_MAX_ACTIVE_SESSIONS:-16} \
VIME_MEM_PROBE=1 FEAT_TRAIN_EXPANDABLE=1 VIME_EMPTY_CACHE_PER_STEP=1 \
FEAT_PREFIX_CACHE=1 FEAT_MULTISTREAM_SHARED_EXPERT=1 FEAT_STATIC_KERNEL=1 FEAT_HCCL_AIV=1 \
FEAT_LB_PROXY=1 \
bash "${SCRIPT_DIR}/run-qwen36-35b-math.sh"

# ── 看什么(判据)──────────────────────────────────────────────────────
#   ✅ reward mean 几十步内趋势上升 + train/tis≈1 + pass@1 上升 = 管线通。
#   ⚠️ std≈0 / zero_std 组占比高 → 难度/方差问题(调 N_SAMPLES,不是修管线)。
#   ⚠️ adv 正常但 reward 平 → 查 advantage 符号 / 轨迹→样本映射 / LR。
#   埋点:math_judge 的 EvalResult.metadata 带 pred/gt/correct/gt_found;
#         gt_found=False = 答案没到 judge(排查 metadata 透传,别当解错)。
