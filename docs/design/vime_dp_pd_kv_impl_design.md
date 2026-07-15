# vime rollout:分布式 DP / PD 分离 / KV 池化 —— 精读核实的实现设计(可进正式开发)

- 状态:**设计(精读核实版)/ 待实现**。取代 [[vime_vllm_native_dp_rollout]] §14-17 的侦察级内容(该文 §0-13 的 DP 选型/决策仍有效)。
- 日期:2026-07-11。方法:三个深读 agent 精读 **vime 内核 / vllm-ascend examples+docs+connector / slime 工作参考**,每条 file:line 核实。
- 关联:[[rollout_ep_weight_sync_vllm_fix]](EP 权重同步已通)、[[vime_resource_layout_and_dp]]、[[router_return_token_ids_passthrough]]

---

## 0. 精读纠正的两处侦察级错误
1. **NIXL 不在 vllm-ascend**:vllm-ascend 的 PD **只有两种 Mooncake connector**——`MooncakeConnectorV1`(D 拉 P)、`MooncakeLayerwiseConnector`(P 推 D)(`vllm_ascend/distributed/kv_transfer/__init__.py:29-55`)。vime 的 `VLLM_NIXL_SIDE_CHANNEL_HOST/PORT`(`vllm_engine.py:477-478`)是上游通用脚手架、**Ascend 上是死 env**(无 `--kv-transfer-config` 消费它)。→ PD 的 KV 走 Mooncake,不是 NIXL。
2. **slime PD 靠 sglang 原生 RDMA**,不搬 KV;vime 换 vLLM 后必须用 `--kv-transfer-config`(Mooncake)显式配 KV 传输。

## 1. vime 对象模型(精读核实)
```
RolloutManager (ray actor, rollout.py:354)  self.servers: dict[name→RolloutServer]
  RolloutServer (1 model = 1 router, rollout.py:214)  server_groups: list[ServerGroup]
    ServerGroup (同构引擎组, 1 worker_type, rollout.py:44)
      worker_type ∈ regular/prefill/decode/placeholder/encoder (:58)
      rank_offset(累计引擎数)/ gpu_offset(累计卡数)
      all_engines: list[VLLMEngine]
        VLLMEngine (ray actor = 1 vllm serve 子进程, vllm_engine.py:548)
```
**引擎创建/分卡/端口唯一入口** = `ServerGroup.start_engines`(rollout.py:76);引擎数 = `num_gpus // per_engine`(rollout.py:1100);`base_gpu_id = reordered_gpu_ids[gpu_offset+i*per_engine]`(:120)→ `visible_devices` 连续段(vllm_engine.py:1162,位置推导)。

## 2. 已 fork 到位(可直接用)vs 需新建(agent C 核实)
| 能力 | slime | vime 现状 | 结论 |
|---|---|---|---|
| 多引擎 DP(③ 副本 + router LB) | ✅ | ✅ **已 fork**(rollout.py:1100 起 N 引擎 + vllm_engine.py:641 注册 router) | **跑 N 副本 = 纯配置**(`ROLLOUT_NUM_GPUS>PER_ENGINE`),仅依赖 router 透传 return_token_ids |
| weight-sync 多引擎/EP 拓扑 | ✅ engine_gpu_counts+累计 offset | ✅ **已保留**(传输换 vLLM `NCCLWeightTransferEngine`),EP 已实测(TIS=1.0018) | 无需改;异构 P/D 组天然支持(docstring update_weight_from_distributed.py:360) |
| resource_layout 钉位 | ✅ | ✅ **逐行相同**(仅 sglang_dp_size→vllm_dp_size);但**尚未接进 rollout 分卡**(vime_resource_layout_and_dp.md §4.5 待实现) | 接线待做 |
| **vLLM 原生 external-LB DP(②)** | (sglang 内部 DP) | ❌ **脚手架**:`--data-parallel-*` 只在 multi_node 拼(vllm_engine.py:181),单机 DP 路径未行使 | **需改 3 处**(§3) |
| **PD 引擎激活** | ✅ `disaggregation_mode`+bootstrap(sglang_engine.py:546-557) | ❌ `worker_type!=regular` 被短路"treated as regular"(vllm_engine.py:616) | **需接线**(§4) |
| **PD 的 KV 传输** | ✅ sglang 原生 RDMA | ❌ 无 `--kv-transfer-config`/connector | **需接 Mooncake connector**(§4) |
| PD router | 动态 `/workers` 注册 | 静态 prefill_urls/decode_urls 注入(rollout.py:1180-1201) | 已 wired,路线不同 |

## 3. 分布式 DP 实现(⚠️ 2026-07-12 按 ref 修正:走 vLLM 原生分布式 DP + proxy,不走"③ 副本捷径")
**✅ 参考路线(ref.md §2.7/§3.2 + ref2 实测,唯一走这条)= vLLM 原生 external-LB 分布式 DP + `dp_load_balance_proxy_server.py`**(请求 token-load 感知 LB,`active_tokens` 最小堆选路、非轮询)。拓扑:ref 定长 DP4TP4 / 变长 DP2TP8;**我们 4 卡预算 = DP2×TP2**(2 个 DP rank、每个 TP2)。每 rank 一个 `vllm serve --data-parallel-size/-rank/-address/-rpc-port --tensor-parallel-size --api-server-count 1`,`launch_online_dp.py` 批量拉起、proxy 前置统一入口。
> 💡 顺带:ref 的 `dp_load_balance_proxy_server.py`(Python HTTP 代理)**可能直接透传 body 里的 `return_token_ids`**,若如此则**用它替代 vime Rust router 可绕开 #3 那个透传坑**——待核实该 proxy 源码是否原样转发 body(#3 是 vime `vllm_router` 的问题,ref proxy 是另一套)。

**❌ ~~③ 多引擎副本捷径~~(非参考,不走)**:vime 多 TP 引擎挂自带 router(`ROLLOUT_NUM_GPUS/PER_ENGINE`,几乎零代码)——但 ref 用的是**原生 DP**(`--data-parallel-*` 协调 DP-attention),不是独立副本。偏离参考,仅原生 DP 短期起不来时作应急、且须标注。

**② vLLM 原生 external-LB DP(= 参考路线)—— 改 3 个函数(agent A 核实)**:
1. `append_vllm_distributed_launch_flags`(vllm_engine.py:181):加 external-LB 分支,拼 `--data-parallel-size/-rank/-address/-rpc-port` + `--data-parallel-external-lb`(现仅 multi_node 拼 `mp` backend)。参考 vllm serve 语义:external-LB 触发 → `api_server_count=1`(每 rank 一 API server,`serve.py:67-89`)。
2. `_allocate_rollout_engine_addr_and_ports_normal`(rollout.py:888):DP 副本组展开 N 槽,rank0 统一分配 dp-address/rpc-port(端口位 `:961` 已留 `30+dp_size`)。
3. `_register_worker_with_router`(vllm_engine.py:641):每个 DP rank 的 API server 各自注册。
- weight-sync **无需改**(per-engine 记账 update_weight_from_distributed.py:367 原样成立)。
- **手动启动参考**(agent B):`launch_online_dp.py --dp-size --tp-size --dp-size-local --dp-rank-start --dp-address --dp-rpc-port` + `dp_load_balance_proxy_server.py --dp-hosts --dp-ports`(请求长度感知 LB)。

**Balance Scheduling(2.5)**:DP 上后可加 `--vllm-additional-config '{"enable_balance_scheduling":true}'`(`patch_balance_schedule.py`),引擎调度层约束各 DP rank 新请求入场,与 proxy 层 LB 互补。

## 4. PD 分离实现(激活脚手架 —— agent A/B/C 核实)
**vime 缺口只有"引擎激活 + KV 接通",其余(worker_type 全链路/端口/router)已 wired。**

激活步骤:
1. **引擎按 worker_type 下发 KV connector**(改 `build_vllm_cmd_and_env` vllm_engine.py:389 或 per-group override):
   - Prefill:`--kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_rank":0,"kv_port":"20001","kv_connector_extra_config":{"prefill":{"dp_size":N_p,"tp_size":T_p},"decode":{"dp_size":N_d,"tp_size":T_d}}}'`。
   - Decode:同 JSON,改 `kv_role":"kv_consumer"`、`kv_rank":1`、`kv_port":"20002"`。
   - (两端 `kv_connector_extra_config` 都要带 P/D 双方并行度;非对称要求 `P_tp % D_tp == 0`,`disaggregated_prefill.md:105`。)
2. **解除短路**:去掉 vllm_engine.py:616 "treated as regular";并解除 `_apply_vllm_overrides` 对 `disaggregation*` 键的主动 skip(:229-231),或另立 kv-transfer 转发白名单。
3. **端口已分**:prefill/decode 各分 `disaggregation_bootstrap_port`(rollout.py:954);Mooncake `kv_port`→`kv_port+num_chips` 会占用,须避开 `[20000,20000+npu/node*1000)`(multi_node guide:225)。
4. **proxy**:pull 用 `load_balance_proxy_server_example.py`(先 P 后 D、把 P 回传的 `kv_transfer_params` 转给 D);vime 已有静态 PD router,需对齐或换用该 proxy。**每节点唯一 `engine_id`**、全局 `PYTHONHASHSEED=0`。
5. **P/D 各自并行度独立**(用户校正:P 也可 DP)——每组独立配 DP/TP/EP;D 侧惯用 DP(§ vllm-ascend 同事)。

## 5. KV 池化实现(⚠️ 2026-07-12 按 ref 修正:走 AscendStoreConnector,不走 OffloadingConnector)
`--vllm-kv-transfer-config` 全局转发**已通**(vllm_engine.py:287,`kv_transfer_config` 是真 vllm arg、不在 SKIPPED_DESTS)。

**✅ 唯一采用 = 参考路线(ref.md §2.8 + ref2 实测部署):`AscendStoreConnector` + backend `mooncake`**。片上 HBM + DRAM 统一池、前缀跨节点可见。imports 在 vllm 0.21.0 干净(只 `vllm.v1.kv_cache_interface`),是 SupportsHMA。**与 PD #5 共 mooncake 依赖 → 编一次 mooncake v0.3.9 两者一起解锁**。配置见脚本 `FEAT_KV_POOL` / ref.md §3.2。
> **我们这版部署要点(2026-07-12 综合官方 `kv_pool.html`/`KV_Cache_Pool_Guide` + 源码核实,与 ref 差异按我们走)**:①连接器键用 **`lookup_rpc_port`**(非 ref 的 `kvpool_rpc_port`——`pool_scheduler.py:671` 只认 lookup_rpc_port/mooncake_rpc_port);②mooncake **v0.3.9** 源码编译(`git clone -b v0.3.9 …Mooncake`→`apt install mpich`→`bash dependencies.sh -y`→`cmake .. -DUSE_ASCEND_DIRECT=ON && make -j && make install`,见 vllm-ascend `pd_disaggregation_mooncake_single_node.md` §Install Mooncake;ref 无 build 步=预装镜像);③`mooncake.json` 我们 backend 读 `protocol:"ascend"`(硬要求)/metadata_server/master_server_address/global_segment_size/local_buffer_size;④env `PYTHONHASHSEED=0`+`MOONCAKE_CONFIG_PATH`+`LD_LIBRARY_PATH→mooncake`;⑤**硬件 A2(910B2C)→ `HCCL_INTRA_ROCE_ENABLE=1`,不用 A3 的 `ASCEND_ENABLE_USE_FABRIC_MEM`/1GB 对齐**(ref 是 A3);⑥**prefix-caching 保持开**(两级命中,代码无互斥;官方 EN 的 --no-enable-prefix-caching 疑误);⑦CANN9.1.0 ✓(官方要 ≥8.5.0);mooncake **v0.3.11+** 才支持 SSD offload(我们 DRAM 池 v0.3.9 够)。

| connector | 结论 |
|---|---|
| **`AscendStoreConnector`(backend mooncake)** | **✅ ref 采用、唯一走这条**;需 mooncake 库+master;0.21.0 imports 干净;SupportsHMA |
| ~~`OffloadingConnector`(NPUOffloadingSpec)~~ | **❌ off-reference 弯路,已废**:实测崩 `ModuleNotFoundError: vllm.v1.kv_offload.abstract`——vllm-ascend↔vllm 0.21.0 kv_offload 模块版本错位(`npu.py` 依赖 `abstract/spec/mediums`,0.21.0 是 `base/factory`)。**ref 根本不用它**;我之前当"轻量入口"是偏离参考、代价是整段弯路 |

⚠️ **GDN-hybrid 通用前置(KV+PD 都吃)**:任何 `--kv-transfer-config` 都触发 vllm 默认关 HMA(`config/vllm.py:1342`),GDN+full-attn 必须 HMA → 加 `--no-vllm-disable-hybrid-kv-cache-manager`(AscendStore 是 SupportsHMA、共存)。ref 是 GLM(DSA、非 GDN-hybrid)故 ref 无此 flag;已验、脚本已内置、PD 复用。

## 6. 分阶段落地(收敛:三者共用 RolloutServer/ServerGroup/worker_type + router + kv-transfer + resource_layout)
- **P-1**:router 方案 A(透传 return_token_ids)—— DP/PD 共用硬前置。
- **P0**:DP-only ③ 多引擎副本(纯配置)过 token-faith 闸;engine/router/layout 三处按 worker_type/role 留位(现成)。
- **P1**:② vLLM 原生 external-LB DP(改 §3 三函数)+ resource_layout 接进分卡。
- **P2**:PD 激活(§4:worker_type→Mooncake connector + 解短路 + proxy);D 侧 DP-replicated。
- **P3**:KV 池化(§5:`AscendStoreConnector`+mooncake,ref 路线;与 PD 共 mooncake 依赖、一起解锁)。
- Balance Scheduling / multistream_overlap_shared_expert 等 additional-config 特性按需叠(见 [[feature_stacking_perf_and_gate]] 总账)。

## 7. 复核:文档↔源码链路(抽查已核)
- DP 三函数扩展点:vllm_engine.py:181 / rollout.py:888 / vllm_engine.py:641 —— agent A 逐一核实存在。
- PD 短路点:vllm_engine.py:616-620 "treated as regular" —— 核实。
- KV 转发通路:vllm_engine.py:287 `_forward_vllm_cli_args` + `kv_transfer_config` 非 skip —— 核实。
- weight-sync per-engine 记账:update_weight_from_distributed.py:367 —— 核实 + EP 实测(TIS=1.0018)。
- Mooncake connector 注册:vllm-ascend `kv_transfer/__init__.py:29-55` —— 核实(NIXL 不存在)。

## 8. connector 实现级核实 + 更正(2026-07-11,agent 精读 vllm-ascend source)

**转发通路确认**:`--vllm-kv-transfer-config '<JSON>'` → `--kv-transfer-config`(vllm_engine.py:287 转发;`kv_transfer_config` 不在 SKIPPED_DESTS,arguments.py:56-77)—— §5 正确。

**❌ OffloadingConnector 路径已废(off-reference,2026-07-12 修正;KV 池化改走 §5 的 AscendStore)。下面两条实测发现保留作记录**:
- 注册在 **vllm core**(非 vllm-ascend):`factory.py:183-187` → `offloading_connector.py:46`;NPU backend `vllm_ascend/kv_offload/npu.py:16 NPUOffloadingSpec`(需 `num_cpu_blocks`)。纯本机 HBM↔CPU-DRAM + LRU + D2H/H2D 双流,无 master/socket。
- **现成 JSON**:`{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"num_cpu_blocks":1000,"block_size":128,"spec_name":"NPUOffloadingSpec","spec_module_path":"vllm_ascend.kv_offload.npu"}}`(无端口无 env;旋钮 `num_cpu_blocks`/`block_size`)。
- caveat:e2e test 标 `skip(deprecated)` 但那指**旧的** native CPUOffloadingConnector;此 `OffloadingConnector+NPUOffloadingSpec` 是替代、user guide 主推 → 单跑冒烟即可。
- ⭐ **HMA 硬约束(2026-07-11 实测发现,KV+PD 通用)**:`--kv-transfer-config` 默认**强关 hybrid KV manager**(vllm `config/vllm.py:1342`:只要设了 kv-transfer 就关、**不认 SupportsHMA**)。但 **GDN-hybrid(GDN+full-attn 两种 KV spec)必须 HMA**,关了引擎 init 崩:`Hybrid KV cache manager is disabled but failed to convert the KV cache specs to one unified type`(首次冒烟实崩)。**OffloadingConnector 是 SupportsHMA(`offloading_connector.py:46`)**→ 加 **`--no-vllm-disable-hybrid-kv-cache-manager`** 保 HMA、连接器与之共存(Ascend 支持 HMA,prefix-cache 一直在用)。⭐ **AscendStoreConnector 与 MooncakeConnector 也都是 SupportsHMA(`mooncake_connector.py:1263`)→ KV 池化(AscendStore)/ PD 在 GDN-hybrid 上同样必须带此 flag**,否则同崩。→ 脚本 `FEAT_KV_POOL` 已内置(commit `5f2d63fb`,实测:warning 0、disable_hybrid=False、过 init 崩点)。
- 🚫 **第二堵墙(2026-07-11 实测):HMA flag 解了后,`OffloadingConnector` 创建时崩 `ModuleNotFoundError: vllm.v1.kv_offload.abstract`**。vllm-ascend `kv_offload/npu.py:7,9,10` 依赖 `vllm.v1.kv_offload.{abstract,mediums,spec}`,但装的 **vllm 0.21.0 的 `vllm/v1/kv_offload/` 是 `base.py/factory.py/cpu/worker/`(模块重构过)**。→ **vllm-ascend ↔ vllm 0.21.0 的 kv_offload 后端版本错位**,非 config 可解,需版本对齐(用户/环境;patch vllm-ascend 多处 import+API 高风险低价值,不自主做)。**结论:废掉 OffloadingConnector 这条;KV 池化走 §5 的 AscendStore(imports 在 0.21.0 干净、只需 mooncake = 与 PD #5 同依赖,编一次 mooncake 两者一起解锁)。HMA flag 是真前置(已验已提交,PD 复用)**。

**⚠️ PD(Mooncake)—— 更正 + 阻塞**:
- **端口 bug 更正**:§4.1 例 `kv_port` 20001/20002 **落在 AscendDirectTransport RDMA 保留区 `[20000, 20000+npu/node*1000)`**(mooncake 多机 guide:227-238)→ `zmq Address already in use` 抖动。**8-NPU/A2:kv_port≥28000(P=28000-01,D=28100-01);16-NPU/A3:≥36000**。
- **现成 JSON**(单机 1P1D,P=TP2/D=TP2 对称):
  - Prefill(producer):`{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":"28000","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":2},"decode":{"dp_size":1,"tp_size":2}}}`
  - Decode(consumer):同,改 `"kv_role":"kv_consumer"` + `"kv_port":"28100"`。两端都带 P/D 双方并行度。
- **`kv_rank` inert**:Mooncake 只认 `kv_role`,从不读 `kv_rank`(全仓 0 命中)→ §4.1 的 `kv_rank:0/1` 无害无效、可省;`engine_id` 省=默认 uuid4;`PYTHONHASHSEED=0` 在 vllm-ascend 无引用(proxy 侧细节、非 connector 需求)。
- 🚫 **阻塞:mooncake 库未装**(`import mooncake`→ModuleNotFoundError;`mooncake_transfer_engine.py:16-19` 缺则 ImportError)→ **PD 起步前须先编 Mooncake v0.3.9(`cmake -DUSE_ASCEND_DIRECT=ON` + LD_LIBRARY_PATH),pd_disaggregation_mooncake_single_node.md:95-142**。OffloadingConnector 无此依赖。
- vime 侧激活改点(不变):解 `vllm_engine.py:616-619` 短路;`disaggregation*` 键仍在 `vllm_engine.py:229` 被 skip → 需转发白名单;PD 两端 `--no-enable-prefix-caching` + proxy `load_balance_proxy_server_example.py`(P-first)。
