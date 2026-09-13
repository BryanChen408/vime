NLAYERS=40
FIRST_K_DENSE_REPLACE=0

FEAT_YARN=${FEAT_YARN:-0}
case "${FEAT_YARN}" in
  0|1) ;;
  *) echo "[yarn][FATAL] FEAT_YARN must be 0 or 1, got ${FEAT_YARN}" >&2; return 1 2>/dev/null || exit 1 ;;
esac

POSITION_EMBEDDING_TYPE=rope
ROTARY_BASE=10000000
ROTARY_PERCENT=0.25
YARN_ARGS=()
QWEN36_VLLM_HF_OVERRIDES='{"architectures":["Qwen3_5MoeForConditionalGeneration"]}'
if [ "${FEAT_YARN}" = "1" ]; then
  POSITION_EMBEDDING_TYPE=yarn
  ROTARY_BASE=${YARN_ROPE_THETA:-10000000}
  ROTARY_PERCENT=${YARN_PARTIAL_ROTARY_FACTOR:-0.25}
  YARN_FACTOR=${YARN_FACTOR:-4.0}
  YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS=${YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS:-262144}
  YARN_BETA_FAST=${YARN_BETA_FAST:-32.0}
  YARN_BETA_SLOW=${YARN_BETA_SLOW:-1.0}
  YARN_MSCALE=${YARN_MSCALE:-1.0}
  YARN_MSCALE_ALL_DIM=${YARN_MSCALE_ALL_DIM:-0.0}
  YARN_CORRECTION_RANGE_ROUND_TO_INT=${YARN_CORRECTION_RANGE_ROUND_TO_INT:-1}
  case "${YARN_CORRECTION_RANGE_ROUND_TO_INT}" in
    1) YARN_CORRECTION_ARG=--yarn-correction-range-round-to-int; YARN_TRUNCATE_JSON=true ;;
    0) YARN_CORRECTION_ARG=--no-yarn-correction-range-round-to-int; YARN_TRUNCATE_JSON=false ;;
    *)
      echo "[yarn][FATAL] YARN_CORRECTION_RANGE_ROUND_TO_INT must be 0 or 1, got ${YARN_CORRECTION_RANGE_ROUND_TO_INT}" >&2
      return 1 2>/dev/null || exit 1
      ;;
  esac
  YARN_ARGS=(
    --rotary-scaling-factor "${YARN_FACTOR}"
    --yarn-original-max-position-embeddings "${YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}"
    --yarn-beta-fast "${YARN_BETA_FAST}"
    --yarn-beta-slow "${YARN_BETA_SLOW}"
    --mscale "${YARN_MSCALE}"
    --mscale-all-dim "${YARN_MSCALE_ALL_DIM}"
    "${YARN_CORRECTION_ARG}"
  )
  printf -v QWEN36_VLLM_HF_OVERRIDES \
    '{"architectures":["Qwen3_5MoeForConditionalGeneration"],"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":%s,"partial_rotary_factor":%s,"factor":%s,"original_max_position_embeddings":%s,"beta_fast":%s,"beta_slow":%s,"mscale":%s,"mscale_all_dim":%s,"truncate":%s}}}' \
    "${ROTARY_BASE}" "${ROTARY_PERCENT}" "${YARN_FACTOR}" \
    "${YARN_ORIGINAL_MAX_POSITION_EMBEDDINGS}" "${YARN_BETA_FAST}" "${YARN_BETA_SLOW}" \
    "${YARN_MSCALE}" "${YARN_MSCALE_ALL_DIM}" "${YARN_TRUNCATE_JSON}"
fi

arr=()
for ((i=0; i<NLAYERS; i++)); do
  if (( i < FIRST_K_DENSE_REPLACE )); then
    arr+=(0)
  else
    arr+=(1)
  fi
done

printf -v MOE_LAYER_FREQ "[%s]" "$(IFS=', '; echo "${arr[*]}")"


MODEL_ARGS=(
   --spec "vime_plugins.models.qwen3_5" "get_qwen3_5_spec"

   --disable-bias-linear
   --qk-layernorm
   --group-query-attention
   --num-attention-heads 16
   --num-query-groups 2
   --kv-channels 256
   --num-layers 40
   --hidden-size 2048
   --ffn-hidden-size 512
   --use-gated-attention

   --normalization RMSNorm
   --apply-layernorm-1p
   --position-embedding-type "${POSITION_EMBEDDING_TYPE}"
   --norm-epsilon 1e-6
   --rotary-percent "${ROTARY_PERCENT}"
   --swiglu
   --untie-embeddings-and-output-weights
   --vocab-size 248320

   --rotary-base "${ROTARY_BASE}"
   ${YARN_ARGS[@]+"${YARN_ARGS[@]}"}

   # qwen3.6 GDN / linear attention
   --linear-key-head-dim 128
   --linear-value-head-dim 128
   --linear-num-key-heads 16
   --linear-num-value-heads 32
   --linear-conv-kernel-dim 4

   # moe
   --moe-ffn-hidden-size 512
   --moe-shared-expert-intermediate-size 512
   --moe-router-score-function softmax
   --moe-token-dispatcher-type alltoall
   --moe-router-topk 8
   --moe-layer-freq "$MOE_LAYER_FREQ"
   --num-experts 256
   --moe-grouped-gemm
   --moe-token-drop-policy probs
   --moe-router-dtype fp32
   # [复核-A 回退 2026-07-14] vime 全局跑 --optimization-level 0(见 run-*.sh:97),
   #   而 MindSpeed FusedMoEPermuteFeature 是 optimization_level=2:opt-level 0 时
   #   is_need_apply=False → 不注册 pre_register_patches 的 dummy-TE →
   #   Megatron transformer_config.py:1810 硬 raise "fused permutation is not available. TE>=2.1.0"。
   #   slime 能开是因为它跑 MindSpeed 默认 opt-level 2(arguments.py:415 default=2)。
   #   要在 vime 开此特性需整体上到 opt-level 2(blast radius,单列待决)。此处保持关闭。
   --no-moe-permute-fusion
   --moe-aux-loss-coeff 0

   # qwen3.5 specific
   --attention-output-gate
   --moe-shared-expert-gate
)
