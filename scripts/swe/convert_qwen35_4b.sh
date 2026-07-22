#!/bin/bash
# ═══ Qwen3.5-4B(SWE 真实目标)HF → Megatron torch_dist 转换 ═══
#   派生自 8B 落地脚本 math_val_wt/convert_qwen3_8b.sh(vime 自己的 convert 工具;slime 的缺 TE 会崩)。
#   逐字照抄 NPU env,只改两处:① MODEL_ARGS 换 qwen3.5-4B.sh  ② 路径换 Qwen3.5-4B。
#
#   Qwen3.5-4B = VLM + hybrid GatedDeltaNet(HF `Qwen3_5ForConditionalGeneration`)。vime 有完整正向支持:
#     · vime_plugins/mbridge/qwen3_5.py(hf↔megatron 权重映射)+ vime_plugins/models/qwen3_5.py(get_qwen3_5_spec)
#     · tools/convert_hf_to_torch_dist.py 已 `import vime_plugins.mbridge.qwen3_5` + AutoBridge + 处理 VLM text_config
#     · MindSpeed 有 GDN 算子(chunk_gated_delta_rule / causal_conv1d)供后续训练
#   换 4B 的动机:治掉 8B-instruct 的 hermes tool-call JSON 报错(4B 用 qwen3_coder XML,不需转义)+ 完全对齐同事。
#
#   前置(缺一不可,脚本会防呆):
#     ① 已下载:modelscope download --model Qwen/Qwen3.5-4B --local_dir /home/docker/Qwen3.5-4B
#     ② 磁盘腾出 ≥20G(torch_dist 约 16G;当前盘 100% 满,须先清)
#
#   ⚠️ 关键(实测踩过):torchrun 命令必须带 --qwen-gdn-backend npu —— qwen3_5 的 GDN Attention 默认
#      backend='fla'(需 flash-linear-attention),NPU 没装 fla 会崩(ImportError: No module named 'fla')。
#      npu backend 走 MindSpeed 算子(chunk_gated_delta_rule/causal_conv1d)。参考 35B 训练脚本同样带此 arg。
#      (--use-gated-attention:vime Megatron 认得 arguments.py:1774,不报错,无需删。)
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -ex
VIME_DIR=${VIME_DIR:-/workspace/vime}
export TORCH_DEVICE_BACKEND_AUTOLOAD=0          # NPU convert 必须(autoload 坑,见 memory convert-tool-slime-vs-vime-npu)
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-8,9,10,11}
_NODE_IPS=$(hostname -I 2>/dev/null | tr ' ' ',' | sed 's/,\+$//')
export no_proxy="127.0.0.1,localhost,${_NODE_IPS}"
export NO_PROXY="${no_proxy}"
HF_PATH=${HF_PATH:-/home/docker/Qwen3.5-4B}
SAVE_PATH=${SAVE_PATH:-/home/docker/Qwen3.5-4B_torch_dist}

# ── 前置防呆(缺权重 / 磁盘不够就别白占卡)────────────────────────────────
if [ ! -f "${HF_PATH}/config.json" ]; then
   echo "[FATAL] 缺 ${HF_PATH}/config.json —— 先下载 Qwen/Qwen3.5-4B 到 ${HF_PATH}" >&2
   exit 1
fi
AVAIL_G=$(df -BG "${SAVE_PATH%/*}" 2>/dev/null | awk 'NR==2 {gsub(/[^0-9]/,"",$4); print $4+0}')
if [ "${AVAIL_G:-0}" -lt 20 ]; then
   echo "[FATAL] $(dirname "${SAVE_PATH}") 仅剩 ${AVAIL_G}G,torch_dist 需 ~16G —— 先清磁盘再来" >&2
   exit 1
fi

rm -rf "${SAVE_PATH}"
mkdir -p "${SAVE_PATH}"
cd "${VIME_DIR}"
source scripts/models/qwen3.5-4B.sh             # MODEL_ARGS(vime 现成:qwen3_5 spec + GDN 结构;不改)
PYTHONPATH="/workspace/Megatron-LM:${VIME_DIR}:${PYTHONPATH:-}" \
  # ⚠️ 单进程 PP1 TP1(照同事 convert_weights.sh):4B tied(lm_head=embed)+MTP 在 PP>1 会跨 stage 撞
  #   language_module 的 tie_embeddings_and_output_weights_state_dict assert;单 stage 则不冲突。
  #   (8B 能用 nproc 4 是因为 8B untied 无 MTP;别对 4B 用多进程/多卡分片。)
  torchrun --nproc-per-node "${NPROC:-1}" --master-addr 127.0.0.1 --master-port 29556 \
    tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --qwen-gdn-backend npu \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --hf-checkpoint "${HF_PATH}" \
    --save "${SAVE_PATH}"
echo "=== 转换完成: ${SAVE_PATH} ==="
echo "=== 下一步:start_swegym 换 HF_CHECKPOINT/REF_LOAD 到 4B + tool-parser 换 qwen3_coder(治 hermes) ==="
