# GRPO against the polar operator environment. Start polar first; this only drives vime.
#
# Single node by default, with the layout that keeps the actor inside one HCCS domain.
# The addresses come from SOCKET_IFNAME unless given, so the same file works on any of the
# machines. Override anything from the environment:
#
#   NNODES=2 RESOURCE_LAYOUT=.../resource_layout.dual88train52infer.yaml bash scripts/start.sh
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-4,5,6,7,8,9,10,11,12,13,14,15}" \
SOCKET_IFNAME="${SOCKET_IFNAME:-ens1f3}" \
NNODES="${NNODES:-1}"  NPUS_PER_NODE="${NPUS_PER_NODE:-12}" \
RESOURCE_LAYOUT="${RESOURCE_LAYOUT:-${SCRIPT_DIR}/resource_layout_actor_domain2.yaml}" \
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-32768}" \
VIME_MEM_PROBE="${VIME_MEM_PROBE:-1}" \
FEAT_TRAIN_EXPANDABLE="${FEAT_TRAIN_EXPANDABLE:-1}" \
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"  N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}" \
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"  NUM_ROLLOUT="${NUM_ROLLOUT:-200}" \
FEAT_DP_EXTERNAL_LB=0 FEAT_BALANCE_SCHED=0 FEAT_LB_PROXY=1 FEAT_CROSS_DP_EP=0 \
FEAT_ROLLOUT_EP=0 FEAT_FLASHCOMM1=0 FEAT_PREFIX_CACHE=1 FEAT_MULTISTREAM_SHARED_EXPERT=1 \
FEAT_STATIC_KERNEL=1 FEAT_HCCL_AIV=1 \
bash "${SCRIPT_DIR}/run-qwen36-35b-polar-minimal.sh"
