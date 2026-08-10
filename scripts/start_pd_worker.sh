#!/usr/bin/env bash
# 140 (rollout+agent / ray worker) 启动。
# 用法:先在 141 跑 `bash scripts/start_pd.sh` 等到 active=1/2,再在 140 跑本脚本。
# 只 join ray 集群;引擎与 PD proxy 由 head(141)的 driver 远程创建。
#
# 用显式 export(不用 `VAR=val \` 续行链)—— 续行链一旦某行漏了结尾反斜杠,
# 该变量会静默变成本地未导出变量、传不进内层 bash(实测 ASCEND_RT_VISIBLE_DEVICES 就这么丢过)。
set -euo pipefail

# ── 卡位:worker 暴露 4-7 给 vime(推理4卡,P=4-5/D=6-7 同 HCCS 域);0-3 归 polar agent。
#    必须与 layout 的 rollout devices "4-7" 完全一致,否则 ray 注册的卡与 layout 对不上 → select_role_bundles 报错。
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export NPUS_PER_NODE=4

# ── 网络 / 角色 ──
export CURRENT_IP=80.5.25.140
export MASTER_ADDR=80.5.25.141
export NNODES=2
export SOCKET_IFNAME=enp48s3u1u1     # 140 上承载 80.5.25.140 的网卡;换机改这里

# ── FEAT(与 head 保持一致;worker 的 raylet 环境会被远程引擎继承)──
export FEAT_TRAIN_EXPANDABLE=1
export VIME_EMPTY_CACHE_PER_STEP=1
export FEAT_DP_EXTERNAL_LB=0 FEAT_BALANCE_SCHED=0 FEAT_LB_PROXY=1 FEAT_CROSS_DP_EP=0
export FEAT_ROLLOUT_EP=0 FEAT_FLASHCOMM1=0 FEAT_PREFIX_CACHE=1 FEAT_MULTISTREAM_SHARED_EXPERT=1 FEAT_STATIC_KERNEL=0 FEAT_HCCL_AIV=1

# ── [WSYNC 排障 2026-08-08] 与 head 的 start_pd.sh 同步,覆盖 run 脚本的 HCCL 默认 ──
# 权重同步组跨 8-11/12-15 两个 HCCS 域;INTER_HCCS_DISABLE=true 时跨域只能走 RoCE(实测不通)→ 组 warmup 600s 超时。
export TRANSFORMERS_VERBOSITY=error
export HCCL_INTER_HCCS_DISABLE=false
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_INTRA_PCIE_ENABLE=0
export HCCL_BUFFSIZE=512

# ── [PD mooncake 修复 2026-08-08] adxl/hixl 建链两连坑(详见 memory weight-sync-hang-debug)──
# ① hixl 取 device_ip:先读 /etc/hccn.conf,不存在才回退 hccn_tool。140 上是 0 字节空文件 →
#    空表 → Connect PARAM_INVALID(103900)。两条防线:删空文件 + 保证 hccn_tool 可达。
#    【需配合:mv /etc/hccn.conf /etc/hccn.conf.bak(空文件,无内容可失)】
export PATH="/usr/local/Ascend/driver/tools:${PATH}"
# ② 同机跨域建链实测要 ~10.1s,默认 ASCEND_CONNECT_TIMEOUT=10000 会竞态误判 timeout →
#    调大到 60s(今早 141 的 "Connect timeout" 即此)。
export ASCEND_CONNECT_TIMEOUT=${ASCEND_CONNECT_TIMEOUT:-60000}

cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
exec bash scripts/run-qwen36-35b-polar-multi-pd.sh
