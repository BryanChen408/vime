#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# HCCL 跨机 NPU RoCE 数据面连通性验证(141 训练 16 卡 ↔ 140 rollout 16 卡 = 32 卡)
#
# 背景:权重同步组建 HcclAllReduce 时报 "HCCL error: timeout",怀疑跨机 NPU
#       RoCE 数据面不通。本脚本用 hccn_tool 逐卡实测,定位到底哪张卡/哪个网段断。
#
# 关键约束(实测):
#   • device IP 分两个 plane:偶数卡 10.0.245.x,奇数卡 10.0.244.x。
#     跨机 HCCL 只在【同 /24 网段】内可达 → 脚本只在同网段配对 ping,跨网段跳过。
#   • 同机卡间 RoCE ping 常态 100% 丢包(走 HCCS 不 hairpin),不代表故障 → 只测跨机。
#
# 用法(两台都要能跑本脚本):
#   ① 在【对端】机器上采集它的卡 IP 清单:
#        bash scripts/hccl_net_check.sh collect > /tmp/peer_ips.txt
#      把 /tmp/peer_ips.txt 拷到本机(或用下面的 --ssh 自动拉)。
#   ② 在【本机】跑连通性测试:
#        bash scripts/hccl_net_check.sh test /tmp/peer_ips.txt
#      或(本机能免密 ssh 到对端时,一步到位):
#        bash scripts/hccl_net_check.sh test --ssh root@80.5.25.141
#
# 退出码:0=同网段全通; 1=存在不通的同网段对; 2=用法/环境错误。
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

HCCN=${HCCN:-/usr/local/Ascend/driver/tools/hccn_tool}
NDEV=${NDEV:-16}        # 逻辑卡数(A3:8 模组 ×2 die=16)
PKT=${PKT:-32}          # ping 包长
LABEL=${LABEL:-$(hostname)}

[ -x "$HCCN" ] || { echo "[FATAL] 找不到 hccn_tool: $HCCN" >&2; exit 2; }

# 取某卡 device IP(空=该卡无 IP/不存在)
dev_ip()   { timeout 6 "$HCCN" -i "$1" -ip   -g 2>&1 | sed -n 's/^ipaddr://p' | head -1; }
dev_link() { timeout 6 "$HCCN" -i "$1" -link -g 2>&1 | sed -n 's/^link status: //p' | head -1; }
subnet24() { echo "${1%.*}"; }   # 10.0.245.11 -> 10.0.245

# ── collect:打印本机清单,每行 "dev ip subnet link" ──
cmd_collect() {
   echo "# host=${LABEL} generated=$(date '+%F %T')"
   local i ip lk
   for i in $(seq 0 $((NDEV-1))); do
      ip=$(dev_ip "$i"); [ -n "$ip" ] || continue
      lk=$(dev_link "$i"); lk=${lk:-UNKNOWN}
      echo "$i $ip $(subnet24 "$ip") $lk"
   done
}

# ── ping 一对,回显 OK/FAIL + 丢包率 ──
# 成功判据:received > 0
ping_pair() {
   local dev=$1 rip=$2 out recv loss
   out=$(timeout 20 "$HCCN" -i "$dev" -ping -g address "$rip" pkt "$PKT" 2>&1)
   recv=$(echo "$out" | sed -n 's/.* \([0-9]*\) received,.*/\1/p' | head -1)
   loss=$(echo "$out" | sed -n 's/.*received, \([0-9.]*\)% packet loss.*/\1/p' | head -1)
   recv=${recv:-0}; loss=${loss:-100}
   if [ "$recv" -gt 0 ] 2>/dev/null; then echo "OK loss=${loss}%"; else echo "FAIL loss=${loss}%"; fi
}

cmd_test() {
   local peer_file="" ssh_target=""
   if [ "${1:-}" = "--ssh" ]; then
      ssh_target=${2:-}; [ -n "$ssh_target" ] || { echo "[FATAL] --ssh 需要目标,如 root@80.5.25.141" >&2; exit 2; }
      peer_file=$(mktemp)
      echo "[*] ssh ${ssh_target} 采集对端卡 IP ..." >&2
      ssh -o StrictHostKeyChecking=no "$ssh_target" bash -s collect < "$0" > "$peer_file" \
         || { echo "[FATAL] ssh 采集失败" >&2; exit 2; }
   else
      peer_file=${1:-}
      [ -f "$peer_file" ] || { echo "[FATAL] 对端清单文件不存在: ${peer_file:-<空>}" >&2; exit 2; }
   fi

   local peer_host; peer_host=$(sed -n 's/^# host=\([^ ]*\).*/\1/p' "$peer_file" | head -1)
   echo "============================================================"
   echo " 本机: ${LABEL}    对端: ${peer_host:-?}   (pkt=${PKT})"
   echo " 规则: 仅同 /24 网段配对 ping;received>0 记 OK"
   echo "============================================================"

   # 读对端清单到数组
   local -a R_DEV R_IP R_NET R_LINK
   while read -r d ip net lk; do
      [[ "$d" =~ ^[0-9]+$ ]] || continue
      R_DEV+=("$d"); R_IP+=("$ip"); R_NET+=("$net"); R_LINK+=("$lk")
   done < "$peer_file"
   [ "${#R_IP[@]}" -gt 0 ] || { echo "[FATAL] 对端清单为空/格式不对" >&2; exit 2; }

   local i ip net lk j fails=0 pairs=0 down=0
   for i in $(seq 0 $((NDEV-1))); do
      ip=$(dev_ip "$i"); [ -n "$ip" ] || continue
      net=$(subnet24 "$ip"); lk=$(dev_link "$i")
      if [ "$lk" != "UP" ]; then
         printf "  [本机 npu %-2s %s] link=%s  ← 本地网口未 UP,跳过\n" "$i" "$ip" "$lk"; down=$((down+1)); continue
      fi
      for j in "${!R_IP[@]}"; do
         [ "${R_NET[$j]}" = "$net" ] || continue          # 只测同网段
         pairs=$((pairs+1))
         res=$(ping_pair "$i" "${R_IP[$j]}")
         case "$res" in
            OK*) printf "  [npu %-2s %s] -> 对端 npu %-2s %s : %s\n" "$i" "$ip" "${R_DEV[$j]}" "${R_IP[$j]}" "$res" ;;
            *)   printf "  [npu %-2s %s] -> 对端 npu %-2s %s : \033[31m%s\033[0m\n" "$i" "$ip" "${R_DEV[$j]}" "${R_IP[$j]}" "$res"; fails=$((fails+1)) ;;
         esac
      done
   done
   echo "------------------------------------------------------------"
   echo " 同网段配对: ${pairs}   不通: ${fails}   本地未UP: ${down}"
   if [ "$fails" -eq 0 ] && [ "$pairs" -gt 0 ]; then
      echo " ✅ 跨机同网段全通。若权重同步仍超时,查 HCCL 控制面(HCCL_IF_IP/端口/白名单),非数据面。"; return 0
   elif [ "$pairs" -eq 0 ]; then
      echo " ⚠️  没有可测的同网段对(两端网段不重叠?检查 device IP 规划)。"; return 2
   else
      echo " ❌ 存在不通的同网段对(见上红色行)。这就是权重同步组 HcclAllReduce timeout 的数据面根因。"; return 1
   fi
}

case "${1:-}" in
   collect) shift; cmd_collect "$@" ;;
   test)    shift; cmd_test "$@" ;;
   *) echo "用法: $0 collect                       # 在对端机器采集卡IP清单
      $0 test <对端清单文件>            # 本机测跨机连通
      $0 test --ssh root@80.5.25.141   # 免密ssh时一步到位" >&2; exit 2 ;;
esac
