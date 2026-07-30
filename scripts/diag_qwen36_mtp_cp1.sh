#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# [A] MTP label-alignment 诊断 —— CP=1 短 run
#
# 目的:mtp.layers.0 是官方训练好的头,评分正确时 mtp_loss 应该很低(~2-4)。实测各 CP4 run
#   却卡在 ~12.4(随机基线 ln(248320))甚至更高 → MTP 头预测与 label 没对齐。本脚本把 CP 关到 1,
#   跑 3 步看 mtp_loss:
#     · CP=1 低(~2-4)、CP4 ≈随机  → bug 在 CP 的 roll/顺序(_roll_tensor_packed_seq / zigzag 对齐)
#     · CP=1 也 ≈随机             → 基础 off-by-one(chunked_mtp_ce label 切片 / roll 方向 / head 未真正参与)
#   这是分清"CP 专属"还是"通用 label" bug 的决定性一步。
#
# 内存:CP=1 时整条序列落在单卡(CP4 时是 1/4),故把 response / model-len / token 预算收窄,
#   避免 [T,vocab] logits OOM。label 对齐 bug 与序列长短无关,短序列一样能暴露。
#
# 拓扑:关掉 CP 后 actor 的 model-parallel = TP2×CP1 = 2,8 张卡 → dense DP 变 4(CP4 时是 DP1)。
#   EP 保持 8(专家并行独立于 dense DP,expert-tensor-parallel=1)。若 Megatron 报 EP/DP folding
#   相关错误,先把 EP 调小(export EP=4 再跑)再排查。
#
# 用法:  bash scripts/diag_qwen36_mtp_cp1.sh
#   所有值都可再用环境变量覆盖(下面用 :- 保留覆盖能力)。
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

export CP=${CP:-1}                                  # ← 诊断核心:关掉 Context-Parallel
export MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-4096}   # 单卡整条序列的打包预算
export ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-2048}  # 收窄 response,保证整条 ≤ 预算
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}   # prompt+response ≤ 4096
export NUM_ROLLOUT=${NUM_ROLLOUT:-3}                # 3 步足够读出 mtp_loss 量级
export RUN_ROOT=${RUN_ROOT:-/workspace/vime/runs/diag_mtp_cp1_$(date +%Y%m%d_%H%M%S)}

echo "[diag] CP=${CP} MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU} RESP=${ROLLOUT_MAX_RESPONSE_LEN} NUM_ROLLOUT=${NUM_ROLLOUT}"
echo "[diag] 看 ${RUN_ROOT}/run.log 里的 train/mtp_loss:低(~2-4)=对齐正常;≈12.4=随机=有对齐 bug"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${HERE}/run_qwen36_35b_a3b_dapo_math_mtp_async_npu.sh" "$@"
