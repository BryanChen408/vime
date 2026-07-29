# PPO against the polar operator environment. Start polar first; this only drives vime.
#
# Same shape as start.sh, with the critic's topology and batch sizes. The actor and critic
# take turns on cards 8-15 through CPU offload, rollout holds 4-7, and polar keeps 0-3.
# The layout pins the actor inside one HCCS domain so its collectives stay off inter-domain
# RoCE. Addresses come from SOCKET_IFNAME unless given.
#
# GLOBAL_BATCH_SIZE has to equal ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT, otherwise
# train_iters lands on zero and the scheduler refuses to build.
#
# The rollout traces here run long, and the adapter drops anything over MAX_TOKENS_PER_GPU
# times the CP size, so leaving it at its default silently discards every trace and the run
# never progresses.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

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
bash "${SCRIPT_DIR}/run-qwen36-35b-polar-ppo.sh"
