// Raw-ADXL sleep/wake staleness repro — bypasses Mooncake entirely.
//
// Purpose: decide whether the stale VA->PA NIC binding after a VMM
// unmap/remap cycle lives inside libadxl (CANN, closed-source) or in
// Mooncake's AscendDirectTransport usage.  Outcomes:
//   - reproduces here  -> CANN-layer bug; Mooncake patches cannot fix it;
//                         file an upstream issue with this repro.
//   - clean here        -> Mooncake-side usage bug; a Mooncake patch + rebuild
//                         is the fix path.
//
// Two processes (P on NPU0, D on NPU1, RoCE forced by env):
//   HCCL_INTRA_ROCE_ENABLE=1 ./adxl_raw_remap_probe P 0 <ip> <p_port> <dir>
//   HCCL_INTRA_ROCE_ENABLE=1 ./adxl_raw_remap_probe D 1 <ip> <p_port> <dir>
//
// P: VMM alloc (reserve+mallocPhysical+map, camem-identical) -> RegisterMem ->
//    fill 0xA0 -> signal -> [wait] -> DeregisterMem -> unmap+freePhysical ->
//    mallocPhysical+map SAME VA -> fill 0xB0 -> RegisterMem again -> signal.
// D: register local buf -> Connect -> read -> verify 0xA0 -> [wait] -> read
//    again -> report what arrived (0xB0 = fresh/correct, 0xA0 = stale).
//
// Build: bash build_adxl_raw_remap_probe.sh

#include <acl/acl.h>

#include <chrono>
#include <cstdint>
#include <map>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>

#include "adxl/adxl_engine.h"

static const size_t kSize = 64 * 1024 * 1024;

static const char *g_role = "?";

#define CK(x, what)                                          \
  do {                                                       \
    auto _e = (x);                                           \
    if (_e != 0) {                                           \
      fprintf(stderr, "[%s] %s failed: %d\n", g_role, what,  \
              static_cast<int>(_e));                         \
      exit(1);                                               \
    }                                                        \
  } while (0)

static void publish(const std::string &dir, const char *name) {
  std::ofstream(dir + "/" + name) << "ok\n";
}

static void wait_for(const std::string &dir, const char *name) {
  std::string path = dir + "/" + name;
  for (int i = 0; i < 1200; i++) {
    std::ifstream f(path);
    if (f.good()) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }
  fprintf(stderr, "timeout waiting for %s\n", path.c_str());
  exit(2);
}

// Fill/read the device buffer byte pattern via host staging.
static void fill_byte(void *dev_va, uint8_t byte) {
  void *host = nullptr;
  CK(aclrtMallocHost(&host, kSize), "mallocHost");
  memset(host, byte, kSize);
  CK(aclrtMemcpy(dev_va, kSize, host, kSize, ACL_MEMCPY_HOST_TO_DEVICE), "H2D");
  CK(aclrtFreeHost(host), "freeHost");
}

// Returns the first-byte-distinct summary: 0 = all expected, 1 = all zero,
// 2 = all other byte, 3 = mixed.
static int classify(void *dev_va, uint8_t expected) {
  void *host = nullptr;
  CK(aclrtMallocHost(&host, kSize), "mallocHost");
  CK(aclrtMemcpy(host, kSize, dev_va, kSize, ACL_MEMCPY_DEVICE_TO_HOST), "D2H");
  uint8_t *p = static_cast<uint8_t *>(host);
  size_t match = 0, zero = 0;
  for (size_t i = 0; i < kSize; i++) {
    if (p[i] == expected) match++;
    if (p[i] == 0) zero++;
  }
  aclrtFreeHost(host);
  printf("[D] match=%.4f zero=%.4f expected=0x%02x\n",
         static_cast<double>(match) / kSize, static_cast<double>(zero) / kSize,
         expected);
  if (match == kSize) return 0;
  if (zero == kSize) return 1;
  if (match == 0 && zero == 0) return 2;
  return 3;
}

int main(int argc, char **argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: %s P|D <dev> <ip> <p_port> <dir>\n", argv[0]);
    return 64;
  }
  const char *role = argv[1];
  g_role = role;
  int dev = atoi(argv[2]);
  const char *ip = argv[3];
  int p_port = atoi(argv[4]);
  std::string dir = argv[5];

  CK(aclrtSetDevice(dev), "aclrtSetDevice");

  // VMM alloc identical to vllm-ascend camem my_malloc.
  aclrtPhysicalMemProp prop = {};
  prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
  prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
  prop.memAttr = ACL_HBM_MEM_HUGE;
  prop.location.id = dev;
  prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  prop.reserve = 0;

  void *va = nullptr;
  CK(aclrtReserveMemAddress(&va, kSize, 0, nullptr, 0), "reserve");
  aclrtDrvMemHandle handle = nullptr;
  CK(aclrtMallocPhysical(&handle, kSize, &prop, 0), "mallocPhysical");
  CK(aclrtMapMem(va, kSize, 0, handle, 0), "mapMem");
  printf("[%s] va=%p size=%zu\n", role, va, kSize);

  adxl::AdxlEngine eng;
  char self_name[64];
  int my_port = (role[0] == 'P') ? p_port : p_port + 1;
  snprintf(self_name, sizeof(self_name), "%s:%d", ip, my_port);
  std::map<adxl::AscendString, adxl::AscendString> options;
  options["adxl.BufferPool"] = "0:0";
  CK(eng.Initialize(adxl::AscendString(self_name), options), "adxl Initialize");
  printf("[%s] adxl engine %s up\n", role, self_name);

  adxl::MemHandle mem_handle = nullptr;
  adxl::MemDesc md{};
  md.addr = reinterpret_cast<uintptr_t>(va);
  md.len = kSize;
  CK(eng.RegisterMem(md, adxl::MEM_DEVICE, mem_handle), "RegisterMem");
  printf("[%s] RegisterMem ok handle=%p\n", role, mem_handle);
  fflush(stdout);
  publish(dir, "engine_up");

  char peer_name[64];
  snprintf(peer_name, sizeof(peer_name), "%s:%d", ip, p_port);

  if (role[0] == 'P') {
    fill_byte(va, 0xA0);
    { std::ofstream(dir + "/p_va") << std::hex << reinterpret_cast<uintptr_t>(va) << "\n"; }
    publish(dir, "p_phase0");
    wait_for(dir, "d_phase0");

    // sleep/wake: unmap + freePhysical, then mallocPhysical + map SAME VA.
    CK(eng.DeregisterMem(mem_handle), "DeregisterMem");
    CK(aclrtUnmapMem(va), "unmapMem");
    CK(aclrtFreePhysical(handle), "freePhysical");
    handle = nullptr;
    CK(aclrtMallocPhysical(&handle, kSize, &prop, 0), "mallocPhysical#2");
    CK(aclrtMapMem(va, kSize, 0, handle, 0), "mapMem#2");
    fill_byte(va, 0xB0);
    CK(eng.RegisterMem(md, adxl::MEM_DEVICE, mem_handle), "RegisterMem#2");
    publish(dir, "p_phase1");
    wait_for(dir, "d_phase1");

    // phase2: D remaps its LOCAL (destination) buffer.  Keep the connection
    // story identical: D disconnects, deregisters, remaps same VA, re-fills
    // 0xEE as a sentinel, re-registers; we then refill 0xC0 and D reads.
    wait_for(dir, "d_phase2_ready");
    fill_byte(va, 0xC0);
    publish(dir, "p_phase2");
    wait_for(dir, "d_phase2");
    printf("[P] done\n");
  } else {
    wait_for(dir, "engine_up");
    adxl::Status cst = adxl::FAILED;
    for (int i = 0; i < 5 && cst != adxl::SUCCESS; i++) {
      cst = eng.Connect(adxl::AscendString(peer_name), 10000);
      if (cst != adxl::SUCCESS) {
        printf("[D] Connect attempt %d failed: %u, retrying\n", i, cst);
        fflush(stdout);
        std::this_thread::sleep_for(std::chrono::seconds(3));
      }
    }
    if (cst != adxl::SUCCESS) { fprintf(stderr, "[D] Connect failed permanently\n"); exit(1); }
    wait_for(dir, "p_phase0");
    adxl::TransferOpDesc op{};
    op.local_addr = reinterpret_cast<uintptr_t>(va);
    op.remote_addr = 0;  // filled from file below? no: P uses same VA scheme; we need P's va
    // P's va is published via a file to be exact:
    {
      // read P's va
      std::ifstream f(dir + "/p_va");
      std::string s;
      f >> s;
      op.remote_addr = std::stoull(s, nullptr, 16);
    }
    op.len = kSize;
    auto st = eng.TransferSync(adxl::AscendString(peer_name), adxl::READ,
                               std::vector<adxl::TransferOpDesc>{op}, 30000);
    printf("[D] phase0 transfer status=%u\n", st); fflush(stdout);
    int c0 = classify(va, 0xA0);
    // Hypothesis: an open connection pins the peer's registration and makes
    // DeregisterMem return PARAM_INVALID.  Disconnect before signaling P.
    auto dst = eng.Disconnect(adxl::AscendString(peer_name), 5000);
    printf("[D] disconnect status=%u\n", dst); fflush(stdout);
    publish(dir, "d_phase0");

    wait_for(dir, "p_phase1");
    CK(eng.Connect(adxl::AscendString(peer_name), 10000), "Connect#2");
    st = eng.TransferSync(adxl::AscendString(peer_name), adxl::READ,
                          std::vector<adxl::TransferOpDesc>{op}, 30000);
    printf("[D] phase1 transfer status=%u\n", st); fflush(stdout);
    int c1 = classify(va, 0xB0);
    publish(dir, "d_phase1");
    printf("[D] RESULT phase0=%s phase1=%s\n",
           c0 == 0 ? "OK" : "BAD", c1 == 0 ? "FRESH-OK" : (c1 == 1 ? "ZEROS" : "STALE-OR-MIXED"));
    fflush(stdout);

    // ---- phase2: D-side (destination) remap ----
    dst = eng.Disconnect(adxl::AscendString(peer_name), 5000);
    printf("[D] phase2 disconnect status=%u\n", dst); fflush(stdout);
    CK(eng.DeregisterMem(mem_handle), "D DeregisterMem");
    CK(aclrtUnmapMem(va), "D unmapMem");
    CK(aclrtFreePhysical(handle), "D freePhysical");
    handle = nullptr;
    CK(aclrtMallocPhysical(&handle, kSize, &prop, 0), "D mallocPhysical#2");
    CK(aclrtMapMem(va, kSize, 0, handle, 0), "D mapMem#2");
    fill_byte(va, 0xEE);  // sentinel: if the next transfer lands in the OLD
                          // pages, classify still sees 0xEE
    CK(eng.RegisterMem(md, adxl::MEM_DEVICE, mem_handle), "D RegisterMem#2");
    publish(dir, "d_phase2_ready");
    wait_for(dir, "p_phase2");
    CK(eng.Connect(adxl::AscendString(peer_name), 10000), "Connect#3");
    st = eng.TransferSync(adxl::AscendString(peer_name), adxl::READ,
                          std::vector<adxl::TransferOpDesc>{op}, 30000);
    printf("[D] phase2 transfer status=%u\n", st); fflush(stdout);
    int c2 = classify(va, 0xC0);
    publish(dir, "d_phase2");
    printf("[D] RESULT phase2(dst-remap)=%s\n",
           c2 == 0 ? "FRESH-OK" : (c2 == 2 ? "STALE(landed-in-old-pages)" : "OTHER"));
    fflush(stdout);
  }
  eng.Finalize();
  return 0;
}
