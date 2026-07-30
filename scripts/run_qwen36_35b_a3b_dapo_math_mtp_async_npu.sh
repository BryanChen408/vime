#!/bin/bash
# Qwen3.6-35B-A3B (GDN/MoE) DAPO-math RL + MTP training — ASYNC, single node, NPU.
# Same as run_qwen36_35b_a3b_dapo_math_async_npu.sh, but ALSO trains the 1 MTP layer online
# (see MTP_ARGS below): --mtp-num-layers 1 builds the MTP block into the GPTModel,
# --enable-mtp-training feeds mtp_labels + logs train/*mtp_loss, --mtp-loss-scaling-factor
# weights the MTP loss (default 0.2). MTP weights come from Qwen3.6-35B-A3B_torch_dist
# (verified to contain mtp.layers.0.*); the _fused_torch_dist checkpoint has NO MTP tensors.
# MTP training is incompatible with combined-1f1b (model.py:594); this run uses PP=1.
# Derived from run-qwen36-35b-polar-minimal.sh (async skeleton) but with the
# polar validation framework stripped out and the standard RL+math task wired in:
#   - train_async.py (NOT train.py): async asserts `not colocate`, so vime does
#     NOT inject --worker-extension-cls → avoids the NPUWorker.finish_weight_update
#     collision that the sync+colocate path hits on this vllm-ascend.
#   - actor 8 NPUs + rollout 4 NPUs (disaggregated, no colocation).
#   - data = dapo-math-17k, reward = deepscaler (local \boxed{} checker, no sandbox).
#   - default rollout fn (vime.rollout.vllm_rollout.generate_rollout) — no polar hooks.
#   - DAPO = grpo + decoupled clip (0.2/0.28), kl 0, --use-tis for off-policy async.
set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VIME_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
cd "${VIME_ROOT}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# ─── run id / single-node topology ───
RUN_ID=${RUN_ID:-qwen36_dapo_math_mtp_async_$(date +%Y%m%d-%H%M%S)}
MASTER_ADDR=${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}
CURRENT_IP=${CURRENT_IP:-$(hostname -I | awk '{print $1}')}
NNODES=${NNODES:-1}
NPUS_PER_NODE=${NPUS_PER_NODE:-12}                 # actor 8 + rollout 4
RAY_PORT=${RAY_PORT:-6460}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8290}
RAY_TEMP_DIR=${RAY_TEMP_DIR:-/tmp/ray_qwen36_dapo_math_async}

ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}
ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-4}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-4}

# ─── paths (this machine) ───
HF_CKPT=${HF_CKPT:-/mnt/weight/Qwen3.6-35B-A3B}
REF_LOAD=${REF_LOAD:-/mnt/weight/Qwen3.6-35B-A3B_torch_dist}
PROMPT_DATA=${PROMPT_DATA:-/mnt/share/l30055792/datasets/dapo-math-17k.jsonl}
RUN_ROOT=${RUN_ROOT:-${VIME_ROOT}/runs/dapo_math_mtp_async_$(date +%Y%m%d_%H%M%S)}
LOG_FILE=${LOG_FILE:-${RUN_ROOT}/run.log}
mkdir -p "${RUN_ROOT}"

# ─── environment (mirrors polar-minimal; NPU/HCCL/GDN correctness knobs) ───
export PYTHONPATH="/workspace/vllm:/workspace/vllm-ascend:/workspace/Megatron-LM:/workspace/MindSpeed:${VIME_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/op_api/lib/:${LD_LIBRARY_PATH:-}"
# vllm_ascend_C.so (sleep-mode/CaMem allocator) links libvllm_ascend_kernels.so,
# which lives in the vllm-ascend package dir — put it on the path or the vLLM
# worker crashes with init_module=None under --vllm-enable-sleep-mode.
export LD_LIBRARY_PATH="/workspace/vllm-ascend/vllm_ascend:/workspace/vllm-ascend/vllm_ascend/lib:${LD_LIBRARY_PATH}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TASK_QUEUE_ENABLE=0                     # must be 0: =1 makes GDN/ring-attn train NaN
export TORCHDYNAMO_DISABLE=1                   # Ascend inductor get_gpu_type() assert → eager
export QWEN36_CP_MODE=ulysses
export QWEN36_CAUSAL_CONV1D_IMPL=triton
export QWEN36_CHUNK_LMHEAD=1                    # pairs with --chunked-lm-head: chunk LM-head logprob → avoids the logits.float() OOM
export QWEN36_MTP_CE_CHUNK=${QWEN36_MTP_CE_CHUNK:-1024}   # MTP-head CE chunk over seq. Bigger = fewer all_reduces but higher peak; 4096 (~2GB/chunk) OOM'd at 32768, 1024 (~500MB) is the safe/fast balance.
export CPU_AFFINITY_CONF=1                      # NUMA core-binding (perf)
export VLLM_ASCEND_ENABLE_NZ=0                 # must be 0 for RL weight-sync + wake_up
# vllm here is built 2 commits past the v0.21.0 tag → reports 0.21.1.dev2, which
# makes vllm-ascend's vllm_version_is("0.21.0") guard take the wrong (main-only,
# expert_map_manager) import branch. Pin it so it treats vllm as 0.21.0.
export VLLM_VERSION=0.21.0
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export RAY_DEDUP_LOGS=0                         # stream logs (=1 batches duplicate lines → looks like it flushes every few min)
export PYTHONUNBUFFERED=1                       # real-time stdout (polar's PYTHONBUFFERED=16 is a typo/no-op)
export HCCL_OP_EXPANSION_MODE=AIV              # FEAT_HCCL_AIV (from start.sh)
export VIME_EMPTY_CACHE_PER_STEP=1            # empty NPU cache each step — reduces fragmentation/OOM
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
# HCCL
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-600}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-2400}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-512}      # bigger buffer for 35B weight broadcast (actor→rollout sync)
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11}
# in-cluster only — never proxy internal vllm/router/health traffic
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export no_proxy="127.0.0.1,localhost,${MASTER_ADDR},${CURRENT_IP}"
export NO_PROXY="${no_proxy}"

source "${VIME_ROOT}/scripts/models/qwen3.5-35B-A3B.sh"     # → MODEL_ARGS

# ─── arg groups ───
CKPT_ARGS=(
   --hf-checkpoint ${HF_CKPT}
   --ref-load ${REF_LOAD}
   --save ${RUN_ROOT}/ckpt
   --save-interval 20
   --no-save-optim
   --megatron-to-hf-mode raw
   --optimization-level 0
)

TOPO_ARGS=(
   --actor-num-nodes ${ACTOR_NUM_NODES}
   --actor-num-gpus-per-node ${ACTOR_NUM_GPUS_PER_NODE}
   --rollout-num-gpus ${ROLLOUT_NUM_GPUS}
   --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}
   --update-weights-interval 1
)

# math task: default rollout fn + deepscaler reward (NO polar hooks)
ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_DATA}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout ${NUM_ROLLOUT:-50}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE:-4}
   --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT:-8}
   --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN:-16384}
   --rollout-temperature 1.0
   --global-batch-size ${GLOBAL_BATCH_SIZE:-32}
   --balance-data
)

# online MTP-layer training: build (--mtp-num-layers) + update (--enable-mtp-training) the
# draft layer during RL. Loss shows up as train/*mtp_loss.
# DISABLE_MTP=1 → drop the MTP layer entirely (no build, no train) — diagnostic to isolate
# whether the MTP layer's device memory is what tips 32768 CP4 over the HBM ceiling.
if [ "${DISABLE_MTP:-0}" = "1" ]; then
   MTP_ARGS=()
else
   MTP_ARGS=(
      --mtp-num-layers 1
      --enable-mtp-training
      --mtp-loss-scaling-factor 0.2
   )
fi

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.0
   --kl-loss-type low_var_kl
   --entropy-coef 0.0
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis                       # truncated importance sampling — off-policy async correction
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

# actor parallelism aligned to polar-minimal (proven 35B/8-NPU config):
# TP2×CP4=8 shards vocab (TP) + sequence (CP) so the LM-head logprob fits;
# --chunked-lm-head chunks logits.float() (the OOM point). TP1/CP1 was too memory-hungry.
PERF_ARGS=(
   --tensor-model-parallel-size ${TP:-2}
   --pipeline-model-parallel-size ${PP:-1}
   --context-parallel-size ${CP:-4}
   --expert-model-parallel-size ${EP:-8}
   --expert-tensor-parallel-size 1
   --sequence-parallel
   --chunked-lm-head
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   # Packing back to 32768 (was shrunk to 4096 before the MTP OOM fixes): C8 made the
   # chunked-lm-head bypass work with mtp_process=True, and chunked_mtp_ce chunks the MTP head,
   # so both heads' logits peak at [chunk, V/TP] (seq-independent) — 32768 no longer OOMs.
   # log-probs-chunk-size: LM-head logprob is chunked to this many tokens. TOO SMALL is a trap:
   # 64 → 32768/64 = 512 chunks → ~1000 tiny vocab-parallel all_reduces per logprob, ×recompute,
   # serialized by CUDA_DEVICE_MAX_CONNECTIONS=1 → ~17 min/microbatch (collective storm, not a
   # hang). But 2048 (~1GB/chunk CE) OOM'd at 32768 on step 1 (940MB alloc, 500MB free). 512 →
   # 64 chunks, [512,V/TP] fp32 ~254MB, safe peak, still 8× fewer collectives than 64. Tune this
   # + QWEN36_MTP_CE_CHUNK (env above) between speed (larger) and memory (smaller) for your seq len.
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU:-32768}
   --log-probs-chunk-size ${LOG_PROBS_CHUNK_SIZE:-512}
   --seq-length ${SEQ_LENGTH:-32768}
   # FEAT_TRAIN_EXPANDABLE: Ray actors don't inherit parent env → push expandable_segments
   # into the train actor explicitly (helps the LM-head/train-side fragmentation OOM).
   --train-env-vars '{"PYTORCH_NPU_ALLOC_CONF":"expandable_segments:True"}'
)

VLLM_ARGS=(
   --rollout-backend vllm
   --qwen-gdn-backend npu
   --model-name qwen3_5moeforconditionalgeneration
   --vllm-hf-overrides '{"architectures":["Qwen3_5MoeForConditionalGeneration"]}'
   # NOTE: do NOT use --rollout-lb-proxy here. That Python LB proxy only exposes
   # OpenAI routes (/v1/chat/completions, /v1/completions) for the polar agent;
   # the standard math rollout posts to vime's native /inference/v1/generate, which
   # the proxy 404s on. The default Rust vllm-router serves /inference/v1/generate.
   --vllm-weight-sync-mode native
   --no-vllm-weight-sync-packed
   --vllm-gpu-memory-utilization ${VLLM_GPU_MEM_UTIL:-0.8}
   --vllm-max-num-seqs ${VLLM_MAX_NUM_SEQS:-32}
   --vllm-max-model-len ${VLLM_MAX_MODEL_LEN:-18432}
   --vllm-enable-sleep-mode
   # NOTE: keep this JSON space-free — VLLM_ARGS is expanded UNQUOTED at launch, so any space
   # word-splits the arg and vLLM sees a truncated string ("EOF while parsing"). Same reason the
   # --vllm-additional-config / --vllm-hf-overrides above are space-free.
   --vllm-compilation-config '{"cudagraph_capture_sizes":[4,8,12,16,24,32],"cudagraph_mode":"FULL_DECODE_ONLY"}'
   # [B] MTP speculative decoding on the rollout: the trained mtp.layers.0 head drafts tokens,
   # vLLM verifies against the main head. vime forwards this JSON verbatim to vLLM's
   # speculative_config (arguments.py:24) and then emits spec_accept_rate / spec_accept_length
   # per rollout step (ray/rollout.py:_compute_spec_metrics). num_speculative_tokens=3 drives the
   # single MTP head multi-step (draft depth > --mtp-num-layers is allowed). enforce_eager on the
   # draft path per the vllm-ascend graph-mode recipe. Comment out this line to disable spec-decode.
   --vllm-speculative-config '{"method":"mtp","num_speculative_tokens":3,"enforce_eager":true}'
   # inference features you normally enable (start.sh): FEAT_PREFIX_CACHE + FEAT_MULTISTREAM_SHARED_EXPERT + FEAT_STATIC_KERNEL
   --vllm-enable-prefix-caching
   --vllm-enable-chunked-prefill
   --vllm-additional-config '{"multistream_overlap_shared_expert":true,"ascend_compilation_config":{"enable_npugraph_ex":true,"enable_static_kernel":true}}'
   # async is disaggregated (actor & rollout on separate NPUs) → never offload
   --no-offload-train
   --no-offload-rollout
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --use-flash-attn
   --moe-token-dispatcher-type alltoall
   --no-gradient-accumulation-fusion
   --seed ${SEED:-1234}
)

# ─── Ray head (single node) + launch ───
ray stop --force 2>/dev/null || true
rm -rf "${RAY_TEMP_DIR}"
ray start --head --port "${RAY_PORT}" --dashboard-host=0.0.0.0 \
   --node-ip-address="${CURRENT_IP}" --dashboard-port="${RAY_DASHBOARD_PORT}" \
   --num-gpus="${NPUS_PER_NODE}" --resources='{"NPU": '"${NPUS_PER_NODE}"'}' \
   --temp-dir="${RAY_TEMP_DIR}" --disable-usage-stats

# hand device pinning to Ray (per-actor), as in polar-minimal
unset ASCEND_RT_VISIBLE_DEVICES

python3 train_async.py \
   ${TOPO_ARGS[@]} ${MODEL_ARGS[@]} ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} ${GRPO_ARGS[@]} ${MTP_ARGS[@]} ${PERF_ARGS[@]} ${VLLM_ARGS[@]} \
   ${MISC_ARGS[@]} ${CKPT_ARGS[@]} \
   2>&1 | tee "${LOG_FILE}"

echo "RUN_ROOT=${RUN_ROOT}"
