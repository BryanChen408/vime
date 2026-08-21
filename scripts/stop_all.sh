#!/usr/bin/env bash
# stop_all.sh — 结束当前 run 并立即清理两节点残留(不用等下次启动)。
# 顺序不可换:先 ray stop(此时卡上 vllm 按定义全是残留),再按卡号交集清扫;
# 判定与 runner 的 cleanup_rollout_residue 一致(env 的 ASCEND_RT_VISIBLE_DEVICES 与本机卡集求交),
# 避免误杀 polar 工具池(0-3)与 ray::IDLE 预启动 worker(env 全卡但那也是我们要清的,见下)。
# 用法(.56 上):bash scripts/stop_all.sh
set -u

sweep() {  # $1=逗号分隔卡集
  local devs="$1" pid env_devs overlap pass hit
  for pass in TERM KILL; do
    hit=0
    for pid in $(pgrep -fi "vllm|pd_mooncake_proxy|dp_load_balance_proxy|MegatronTrain|train_async\.py|train\.py" 2>/dev/null); do
      env_devs=$(tr '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null | sed -n 's/^ASCEND_RT_VISIBLE_DEVICES=//p' | head -1)
      [ -n "$env_devs" ] || continue
      overlap=$(printf '%s\n%s\n' "$devs" "$env_devs" | tr ',' '\n' | grep -E '^[0-9]+$' | sort -n | uniq -d | head -1)
      [ -n "$overlap" ] || continue
      echo "[stop] kill -$pass pid=$pid cards=[$env_devs] $(ps -o comm= -p "$pid" 2>/dev/null)"
      kill -"$pass" "$pid" 2>/dev/null || true
      hit=1
    done
    [ "$hit" = 0 ] && break
    [ "$pass" = TERM ] && sleep 6
  done
}

echo "[stop] .56: ray stop --force"
ray stop --force 2>/dev/null
sweep "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"

if ssh -o BatchMode=yes -o ConnectTimeout=4 root@80.48.5.64 true 2>/dev/null; then
  echo "[stop] .64: ray stop --force + sweep(4-15)"
  ssh -o BatchMode=yes root@80.48.5.64 'ray stop --force 2>/dev/null
for pass in TERM KILL; do
  for pid in $(pgrep -fi "vllm|pd_mooncake_proxy|dp_load_balance_proxy"); do
    env_devs=$(tr "\0" "\n" < /proc/$pid/environ 2>/dev/null | sed -n "s/^ASCEND_RT_VISIBLE_DEVICES=//p" | head -1)
    case ",$env_devs," in *",4,"*|*",5,"*|*",6,"*|*",7,"*|*",8,"*|*",9,"*|*",10,"*|*",11,"*|*",12,"*|*",13,"*|*",14,"*|*",15,"*)
      echo "[.64] kill -$pass pid=$pid cards=[$env_devs]"; kill -"$pass" "$pid" 2>/dev/null || true;;
    esac
  done
  [ "$pass" = TERM ] && sleep 6
done
echo "[.64] sweep done"'
else
  echo "[stop] ssh .64 不通 —— 请在 .64 手工执行:"
  echo "  ray stop --force && for p in \$(pgrep -fi 'vllm|pd_mooncake_proxy|dp_load_balance_proxy'); do kill -9 \$p; done"
fi
echo "[stop] 完成。建议两机 npu-smi info 确认无进程占卡后再启新 run。"
