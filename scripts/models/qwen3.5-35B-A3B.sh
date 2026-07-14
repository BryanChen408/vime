NLAYERS=40
FIRST_K_DENSE_REPLACE=0

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
   --position-embedding-type rope
   --norm-epsilon 1e-6
   --rotary-percent 0.25
   --swiglu
   --untie-embeddings-and-output-weights
   --vocab-size 248320

   --rotary-base 10000000

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
