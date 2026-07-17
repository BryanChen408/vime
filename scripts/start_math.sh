#!/bin/bash
# ═══ 一键拉起 math agentic 管线验证(DAPO-17k 走 polar claudecode + math_judge)═══
#
# 目的:验证 polar 接入整条 RL 管线是否正确(轨迹捕获→rebuild→reward→advantage→权重回灌→
#   策略改善),以 reward 训得上去 + TIS≈1 为判据。不是训模型,是给管线冒烟。
#
# ── 前置(polar 侧,一次性)──────────────────────────────────────────────
#   polar 必须运行【含 math_judge 代码】的版本,否则 evaluator strategy=math_judge 解析失败。
#   代码在隔离分支 feature/math-judge @ worktree /home/docker/math_polar_wt。两种上法:
#     A) 让宿主机 polar 从 worktree 起(PYTHONPATH=/home/docker/math_polar_wt/src),或
#     B) 把 3 个文件同步进 polar 运行 checkout 后重启:
#          src/polar/trajectory/evaluator/math_judge.py (新)
#          src/polar/trajectory/evaluator/__init__.py    (+MathJudgeEvaluator)
#          src/polar/trajectory/registry.py              (+register math_judge)
#   重启走 hostctl(见 polar_ctl.py restart_polar_stack),重启后 polar 仍在 :8001。
#   ⚠️ 这会顶掉当前 operator run 的 polar 配置(math 验证期间独占 polar)。
#
# ── 拉起(vime 侧)───────────────────────────────────────────────────────
#   在本 worktree(feature/math-pipeline-validation)执行本脚本即可。
#
set -e
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# 数据:DAPO-17k 走 operator_samples —— prompt 保持消息列表(vime Dataset 要 list;
#   bridge prompt_to_instruction_text 转成字符串 instruction 给 agent);prep 补
#   metadata.op_name(operator_samples 硬需)+ metadata.answer(judge-only,→ math_judge)。
RAW_DAPO=${RAW_DAPO:-/home/docker/datasets/dapo-math-17k.jsonl}
export OPERATOR_TASK_JSONL=${OPERATOR_TASK_JSONL:-/home/docker/datasets/dapo-math-17k-prep.jsonl}
if [ ! -f "${OPERATOR_TASK_JSONL}" ]; then
   echo "[start_math] prep DAPO (+metadata.op_name/answer, prompt 保持 list) → ${OPERATOR_TASK_JSONL}"
   python3 "${SCRIPT_DIR}/prep_dapo_math.py" "${RAW_DAPO}" "${OPERATOR_TASK_JSONL}"
fi

# 拓扑 / 卡位(单机,rollout 4-7 / actor 8-15)
export MASTER_ADDR=${MASTER_ADDR:-80.48.5.88}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-4,5,6,7,8,9,10,11,12,13,14,15}
export VIME_ROLLOUT_LOW_CARDS=1

# ── 冒烟规模:关键是造出组内 reward 方差(避免 zero-std 假阴性)──
#   bs 8 × N 8 = 64 序列/step,远大于算子冒烟的 2×2;DAPO 已按 RL 难度过滤 → 组内自然有对有错。
MATH_MODE=1 \
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
bash "${SCRIPT_DIR}/run-qwen36-35b-polar-minimal.sh"

# ── 看什么(判据)──────────────────────────────────────────────────────
#   ✅ 管线通:reward mean 几十步内趋势上升 + train/tis≈1 + pass@1 上升。
#   ⚠️ std≈0 / zero_std 组占比高 → 难度/方差问题(调 N_SAMPLES,不是修管线)。
#   ⚠️ adv 正常但 reward 平 → 查 advantage 符号 / 轨迹→样本映射 / LR。
#   埋点:先看 debug rollout 里 trajectory 是否多轮(§轨迹形状);math_judge 的 EvalResult.metadata
#         带 pred/gt/correct/gt_found —— gt_found=False 说明答案没进 metadata(排查,别当解错)。
