#!/bin/bash
INT=${1:-4}
echo "# ts used Anon Shmem driverGAP shmDU (GB) | 说明:driverGAP=used-Anon-Cached-kernel(驱动pinned)"
while true; do
  awk '/^MemTotal:/{t=$2}/^MemAvailable:/{a=$2}/^AnonPages:/{an=$2}/^Cached:/{ca=$2}/^Shmem:/{sh=$2}/^Slab:/{sl=$2}/^PageTables:/{pt=$2}/^KernelStack:/{ks=$2}/^VmallocUsed:/{vm=$2}/^Percpu:/{pc=$2}
    END{used=(t-a)/1048576; anG=an/1048576; caG=ca/1048576; shG=sh/1048576; ker=(sl+pt+ks+vm+pc)/1048576; gap=used-anG-shG-ker;
        printf "[%s] used=%.0f Anon=%.0f Shmem=%.0f driverGAP=%.0f", strftime("%H:%M:%S"), used, anG, shG, gap}' /proc/meminfo
  shdu=$(du -sBG /dev/shm 2>/dev/null | awk '{print $1}')
  echo " shmDU=$shdu"
  sleep "$INT"
done
