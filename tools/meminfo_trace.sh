#!/bin/bash
# 每 N 秒记录 /proc/meminfo 关键项(内核全内存账,加总=total)+ top3 进程 RSS
INT=${1:-3}
echo "# ts | used AnonPages Shmem Slab PageTables Mapped KernelStack Vmalloc Cached (GB) | top3-rss"
while true; do
  awk '
    /^MemTotal:/{t=$2} /^MemAvailable:/{a=$2}
    /^AnonPages:/{an=$2} /^Shmem:/{sh=$2} /^Slab:/{sl=$2}
    /^PageTables:/{pt=$2} /^Mapped:/{mp=$2} /^KernelStack:/{ks=$2}
    /^VmallocUsed:/{vm=$2} /^Cached:/{ca=$2}
    END{printf "used=%.0f Anon=%.0f Shmem=%.0f Slab=%.0f PgTbl=%.0f Mapped=%.0f KStk=%.1f Vmalloc=%.0f Cached=%.0f",
        (t-a)/1048576, an/1048576, sh/1048576, sl/1048576, pt/1048576, mp/1048576, ks/1048576, vm/1048576, ca/1048576}' /proc/meminfo
  top=$(ps -eo rss,comm --sort=-rss 2>/dev/null | awk 'NR>=2 && NR<=4{printf " %s:%.0fG", $2, $1/1048576}')
  echo " [$(date +%H:%M:%S)] $(awk '/^MemTotal:/{t=$2}/^MemAvailable:/{a=$2}/^AnonPages:/{an=$2}/^Shmem:/{sh=$2}/^Slab:/{sl=$2}/^PageTables:/{pt=$2}/^Mapped:/{mp=$2}/^KernelStack:/{ks=$2}/^VmallocUsed:/{vm=$2}/^Cached:/{ca=$2}END{printf "used=%.0f Anon=%.0f Shmem=%.0f Slab=%.0f PgTbl=%.0f Mapped=%.0f KStk=%.1f Vmalloc=%.0f Cached=%.0f",(t-a)/1048576,an/1048576,sh/1048576,sl/1048576,pt/1048576,mp/1048576,ks/1048576,vm/1048576,ca/1048576}' /proc/meminfo) |$top"
  sleep "$INT"
done
