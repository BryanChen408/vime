# HANDOFF — PD 分离(prefill/decode disaggregation)重启交接【自洽版】

> 写于 2026-07-15。给**新 session** 重启 PD。**本文自洽,不依赖 memory**(相关 memory 内容已内联)。
> 约定:【实测】真机跑出;【git/代码事实】有 commit/file:line;【推论/待验】没跑过。

---

## 0. 30 秒定位

- **正确分支 = `PD-dev @ ec21b664`**(含全部 PD 修复)。`feature/pd-mooncake @ a8fea1d5` 是旧血统只有 W1-W4 无修复,**别用**。
- 上手:**分支 `PD-dev` 存在但没有 worktree(必须先创建)**:`git worktree add /home/docker/pd_dev_wt PD-dev`(在 `/workspace/vime` 里执行)。若在全新机器(如 52)且本地 git 无 PD-dev,先 `git fetch /home/docker/pd_transfer/PD-dev.bundle PD-dev:PD-dev`(bundle 见 §11)。
- **连接器 = `MooncakeConnectorV1`**(`/workspace/vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`,`is_mamba_group :665`/MambaSpec P==D `:699`)。**不是** `MooncakeHybridConnector`(那是池化用、会撞 scheduler 单组 gap)。fix `cadfcee3`。
- **目标**:单机 1P1D→2P+2D,qwen3.6-35B-A3B GDN-hybrid。
- **权威设计**:`docs/design/pd_disaggregation_dev_plan.md`(极详 code-backed,先读;**2026-07-15 已从游离副本抢救进本 repo**——原始 `/home/docker/vime-polar/docs/design/` 那份**不在任何 git 分支、易随机器重装丢失**)。
- **卡点**:PD 07-13 挂起,死在 D 侧 KV 拉取(见 §3)。重启=啃这个 bug。

## 1. 环境/仓/卡/端口地图(内联自 repo-map)

- 系统:qwen3.6-35B-A3B 异步解耦 RL。**vime**(fork slime,Megatron 训练)+ **polar**(ProRL-Agent-Server,agentic 算子生成)。平台 **Ascend NPU**(非 CUDA)。**EI0013 = 跨 HCCS 域 RoCE 偶发崩**。
- **模型**:40 层 = 30 GDN(linear-attn,Ulysses CP)+ 10 full-attn(ring CP,索引 3,7,…,39,`full_attention_interval=4`)+ MoE 256 experts。→ **GDN-hybrid = HMA 多 KV 组**(线性组 + full-attn 组),PD 的难点全在这。
- **两个 vime checkout**:
  - **LIVE = `/workspace/vime`**(**当前 checkout 在 `feature/lb-proxy` @ 146467d1**;注:旧 memory 说 npu,已变)= 开发真源。
  - clean = `/home/docker/vime-polar/vime`(更旧、已分叉,别当基;但 `docs/design/` 权威文档在它下面)。
- **polar**:`/home/docker/cannbot_debug/ProRL-Agent-Server`(`feat/ascend-smoke`)。**跑在宿主机、ps 不可见、绝不 kill**。前缀合并逻辑 `src/polar/trajectory/builder/prefix_merging.py`。
- **卡位**:polar 0-3;vime 4-15。`VIME_ROLLOUT_LOW_CARDS=1` → rollout 4-7 / actor 8-15。**`ASCEND_RT_VISIBLE_DEVICES` 必须升序**(乱序→torch_npu 见 0 卡崩)。PD 单机 1P1D=两引擎各 TP2 落 rollout 的 4 卡。
- **端口**:vllm **router :8001(吞 return_token_ids,勿直连)**;worker :15000(单引擎时 polar profile 直连此);polar gateway :8100;rollout :8080;observer :18088。

## 2. mooncake build 现状(内联自 transport-split,含 2026-07-12 纠正)

- **当前 build = ADXL**:`cmake .. -DUSE_ASCEND_DIRECT=ON -DUSE_CUDA=OFF -DPython_EXECUTABLE=/usr/local/bin/python3`(离线编成,已 install+import 过)。源在 `/workspace/Mooncake`(v0.3.9)。
- **✅ 关键纠正**:早先"PD=HCCL、与池化 ADXL 互斥、要另编"是**错的**。官方 PD 教程 build 就是 `USE_ASCEND_DIRECT=ON`(ADXL)。**同一个 ADXL build 里 PD 和池化两个连接器都在**:
  - **PD(#5)= `MooncakeConnectorV1`**:走 `mooncake.engine.TransferEngine` P2P,`kv_role=kv_producer/kv_consumer`,**不需要 mooncake_master**。
  - 池化(#6)= `AscendStoreConnector`:走 `mooncake.store` + master。**(#6 已死,别碰,见 §9)**
  - → PD **不用重编 HCCL**,现有 ADXL build 够用。
- **⚠️ LD 陷阱**:必设 `export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/Ascend/cann-9.0.0/python/site-packages/mooncake:$LD_LIBRARY_PATH`,否则 `import mooncake` / `mooncake.engine` 报 `libmooncake_store.so: cannot open shared object file`(库分两处:python 包在 cann site-packages、共享 .so 在 `/usr/local/lib`)。**（若新 session `from mooncake import engine` 报错,先查这个)**。
- 共享库落点:`libmooncake_store.so`/`libtransfer_engine.so`/`ascend_transport.so`/`libmooncake_common.so` → `/usr/local/lib`;python 包 → `/usr/local/Ascend/cann-9.0.0/python/site-packages/mooncake`。
- **ADXL 传输层在 A2 独立验证通**(`scratchpad/adxl_test.py`,protocol=ascend + P2PHANDSHAKE + NPU 设备张量,卡4→6 `transfer_sync_read` 0.08s 逐字节一致)。⚠️ `allocate_managed_buffer` 对 ascend 协议段错误 → 必须 NPU 设备张量,非 host 内存。

## 3. ⛳ PD 停在哪(07-13 挂起定论 —— 重启从这接)

【实测 run e/f/g,PD_DEBUG+GLOG】编排/握手/P 侧**全通**,死在 **D 侧 KV 拉取**:
- ✅ P prefill 全 200,正确填 `kv_transfer_params`(`remote_host`/`remote_port:15002`/`remote_block_ids`(**4 个 HMA 组**:1 full-attn 实块列 + 3 mamba 组多为 0 尾块)/`remote_ptp_size:2`/engine_id);proxy 提取+注入 D 正确(`pd_mooncake_proxy_server.py:934/936`);P 正确 `Delaying free of N blocks` 等 D 拉。
- ❌ **卡死**:D 认出 remote_prefill,`get_num_new_matched_tokens`(`mooncake_connector.py:1433`)空转 30 万+ 次、`num_computed_tokens=0`(`train_qwen36_pd_1p1d_g.log` 那 364MB 全是这一行,已随日志删,结论存此);`update_state_after_alloc` 加入待收,但 **`kv_recv` 的 `transferSync` 永不完成**(`Number of completed receive requests: 0`),**连 GLOG_v=2 都不出 transferSync 的 C++ 日志**(只有 init INFO)。
- **已排除**:prefix-cache(关了仍 hang;#7944 只是必须关非根因)、ADXL 传输层(独立通,§2)、端口(15002 是 params 内容非连接重试)。
- **定论【推论,基于可得日志】**:vllm-ascend 本版本 `MooncakeConnectorV1` 在 **GDN-hybrid PD 的 D 侧拉取有未文档化 bug**(类 upstream #7944)。**最疑 Mamba-state 组的 P→D 传输**——attention KV 是块状能传,Mamba state 定长非块状,可能根本没进传输列表。

## 4. 重启 PD 的攻法(选一/组合)

1. **升 vllm-ascend**:`git branch -a` 有 `origin/docker/upgrade-vllm-v023`(tip:"all2all_utils weight-reload fix",vllm 0.23)——先查它的 mooncake V1 GDN-hybrid PD D 侧是否已修。**版本不匹配是我们很多坑的根**(参考版本 0.23,我们 0.21)。
2. **啃 transferSync 不完成**:D 侧发起 recv 后 C++ 层无日志 → 加 mooncake C++ 埋点/strace,确认"没发起 read"还是"read 挂起";重点核 `mooncake_connector.py` 的 group 遍历 + `is_mamba_group`,**验 Mamba-state 组是否根本没进传输列表**。
3. **换连接器/路径**:若 V1 对 GDN Mamba state 就是不支持,评估上游其它连接器或 layerwise。

**调试基建现成**:`PD_DEBUG=1`(commit `9ae4e538`)开 `VLLM_LOGGING_LEVEL=DEBUG`+`MC_TE_METRIC`+`GLOG_v=2`。第一步验证:**1P1D 差分贪婪(temp=0,PD vs 单引擎逐字比,专验 Mamba-state P→D)**。

## 5. 已建好的脚手架(PD-dev 上,可复现)

| 提交 | 内容 |
|---|---|
| `e17d88f3` | 启动 hook `FEAT_PD_DISAGG=1` → per-engine 压到 2(4卡 1P1D)+ `--prefill-num-servers`(默认1)+ `--disaggregation-backend mooncake`;默认 OFF 零回归。`PD_PREFILL_NUM_SERVERS`/`PD_BACKEND` 可覆盖。2P2D 需 `ROLLOUT_NUM_GPUS=8`+缩 actor |
| `cadfcee3` | 连接器 Hybrid→V1(修 504、P prefill 全 200) |
| `7a280ed0` | BUG1:补 `kv_connector_extra_config` 的 `{"prefill":{"tp_size":N,"dp_size":1},"decode":同}`(init assert 硬需,W2 原注入缺它必崩) |
| W3(`0b999fa3`) | `scripts/pd_mooncake_proxy_server.py`:拷 vllm-ascend `disaggregated_prefill_v1` 示例 + 加 `POST /inference/v1/generate` + `/health` + `build_prefill_request` 适配 GenerateRequest |
| W4(`a8fea1d5` 旧/PD-dev 上有等价) | rollout manager 起 proxy `_start_mooncake_pd_proxy` 替 vllm-router(避 #3) |
| `60026ec2` | per-role 关 prefix-cache(#7944)+ mooncake C++ glog |

**连接器配置(code-backed)**:`kv_role` P=`kv_producer`/D=`kv_consumer`;`kv_port = get_port()` 从 15000 顺分(rollout.py:958,< 20000,避 ADXL RDMA 保留区 `[20000,20000+npu*1000)`);HMA 必开 `--no-vllm-disable-hybrid-kv-cache-manager`;`pcp*dcp==1`(rollout 满足);**P/D 的 TP 必须一致**(`:1738` assert,Mamba 下)。per-role 下发 = `_apply_pd_role_overrides`(`vllm_engine.py`,按 worker_type)。

## 6. 怎么跑 + 验证闸(内联自 validation 命令)

**验证 run 必须对齐完整基线命令,别裸跑脚本默认**(默认为别的场景设,漏 batch/chunk-lmhead/mtpg/low-cards 会崩或不可比,曾连漏 3 次):
```bash
export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/Ascend/cann-9.0.0/python/site-packages/mooncake:$LD_LIBRARY_PATH
PD_DEBUG=1 RUN_ID=qwen36_pd_1p1d_h \
VIME_ROLLOUT_LOW_CARDS=1 \
ROLLOUT_BATCH_SIZE=2 N_SAMPLES_PER_PROMPT=4 GLOBAL_BATCH_SIZE=8 NUM_ROLLOUT=2 \
QWEN36_CHUNK_LMHEAD=1 MAX_TOKENS_PER_GPU=32768 \
FEAT_PD_DISAGG=1 \
bash scripts/run-qwen36-35b-polar-minimal.sh
```
- `VIME_ROLLOUT_LOW_CARDS=1` = rollout→4-7 / actor→8-15(不设则 rollout 落高卡跨 HCCS 域 → EI0013 温床);**不是** `--resource-layout`(那条有 dist-init hang bug)。
- `MAX_TOKENS_PER_GPU=32768`:脚本默认 512 会慢爆,必设。`QWEN36_CHUNK_LMHEAD=1`:长序列防 OOM。`NUM_ROLLOUT=2`=step0+1 闸。
- **token-faith 闸**:`TIS≈1.00` / `train_rollout_logprob_abs_diff≲0.05` / `ois=1` / `ppo_kl=0` / `pg_clipfrac=0`。TIS 逐 token、与 batch 无关,小批即可。
- 崩/停后 relaunch 前**必清残留 EngineCore**(泄漏 ~52GiB 会 OOM);验 `torch.npu.is_available()`(设备释放滞后,只看 npu-smi 不够)。

## 7. polar 端点控制(内联自 hostctl;PD 需切 polar 到 proxy)

- **polar 在宿主机,绝不 kill**。容器内重启/控制走 **hostctl 文件协议**(共享挂载 `/home/docker`,无需网络):
  - root `/home/docker/polar_e2e/hostctl/`;读 token `.../hostctl/token`(动态读别硬编)。
  - 写请求 `requests/{id}.json`(先 .tmp 再 rename):`{"id":<uniq>,"token":<token>,"action":<action>,"args":{}}`;读 `results/{id}.json`。
  - 动作白名单:`status`/**`restart_polar_stack`**/`restart_polar_gateway`/`cleanup_ports`/`tail_logs`。助手 `scratchpad/polar_ctl.py`。restart 20-60s。
  - `restart_polar_stack` 会保持 `base_url: http://80.48.5.88:8001`(复用最新 run 目录,不从默认 profile 重生成)。
- **端点切换**(**验证时才改,现在改会断当前跑**):`profile.vime.yaml` 的 `sglang_router_url` → PD proxy(router_port :8001),**非** worker :15000。**PD 多引擎绕不开编排层,必须经 :8001 proxy**(单引擎才能直连 worker)。

## 8. Landmines(务必先看)

1. **⚠️ 工况 regime(2026-07-15 最新实测血泪)**:这是**突发 agentic 工况**非饱和 serving。实测 **external-LB DP 是 2.3× 净损**(e2e 15→35 当关掉它;干净 A/B)。机制:vLLM DP wave 锁步,空闲 rank 每步被迫跑假前向(`vllm/v1/engine/core.py:1821 execute_dummy_batch`),突发低 duty-cycle → ~3/4 卡空烧。**PD 同为 serving 吞吐优化,收益(prefill/decode 去干扰)只在饱和兑现;对这个 bursty rollout 别默认有吞吐收益**——先饱和度探针 + relative A/B。**token-faith(§6)是硬闸,与收益分开谈**(详见 `feature_stacking_results.md`)。
2. **balance_scheduling 与 PD 互斥**:`vllm-ascend platform.py:620` 要求 `kv_role='kv_both'`;PD 用 producer/consumer → **不能叠 `FEAT_BALANCE_SCHED`**。
3. **#3 router 丢 `return_token_ids`**:vllm-router(0.1.14 预编译 wheel、无源码)反序列化成 typed struct 丢字段 → worker 回 token_ids=null → polar 轨迹长度不符 → 训练饿死。**PD 必须用 W3 的 Python proxy,不是 vllm-router。**
4. **清残留 EngineCore**:relaunch 前必清(泄漏 ~52GiB OOM)。**`pkill -f <pat>` 会匹配到自己命令行自杀**——用显式 PID(`ps|grep|awk '$2~/python/'` 排除自己 bash)或 `[x]` trick(今天又踩一次)。
5. **KV 池化 #6 已死**:GDN×HMA 撞 vLLM `scheduler.py:2102 _update_requests_with_invalid_blocks` 单组假设(`(req_block_ids,)=get_block_ids()`,HMA 2 组 → ValueError too many values to unpack;RFC #36780 closed not-planned)。**PD 用 V1 连接器失败走 raise 不喂 invalid_block_ids,避开该 gap**——别回头做池化。
6. Ascend 铁律:`TASK_QUEUE_ENABLE=0`(否则 GDN 反向 NaN)、`VLLM_ASCEND_ENABLE_NZ=0`(权重回灌冲突)、HMA 必开。

## 9. 分支地图(2026-07-15 核实)

- `feature/lb-proxy`(/workspace/vime 当前,146467d1)= 最近 DP/external-LB 基线。
- **`PD-dev @ ec21b664`** = PD 全量(11 PD 提交 + 修复)+ 顶上 2 个 **parked DP** 提交(`4baf4da6` DP proxy、`ec21b664` DP wiring WIP)。**PD 正体到 `60026ec2` 为止**。
- `feature/pd-mooncake @ a8fea1d5` = 旧 W1-W4 隔离血统,**别用**。
- `pool-dev @ e3f03e9f` = KV 池化停放(已死)。`npu @ f1c53f7f` = 另一基线。

## 10. 参考

- `ref.md`(`/home/docker/ref.md`)§2.8 KV/PD 连接器配置;官方 `pd_disaggregation_mooncake_single_node.md`。**ref 是 GLM-5 饱和 serving 压测(concurrency16/rate0/TPOT≤50ms)——配置抄,收益预期别抄**(见 §8.1)。参考版本 vllm 0.23 / 我们 0.21。
- 权威设计:`docs/design/pd_disaggregation_dev_plan.md` + `docs/design/vime_dp_pd_kv_impl_design.md`(均已抢救进本 repo;后者 §0.1 "只有 V1/Layerwise 两种"是**错的**)。
- 同类 handoff:`/home/docker/vime-polar/docs/design/HANDOFF_cross_engine_EP_and_distributed_DP.md`。
- 遗漏/修复清单:`/home/docker/vime-polar/vime_polar_findings_and_fixes.md`。

## 11. 全新机器(如 52)PD 源码补齐步骤

**背景**:PD 相关内容散落在多处、且部分**不在 git**,新机器/新 checkout 会缺。传输包已备好在 `/home/docker/pd_transfer/`(`PD-dev.bundle` 14MB + `pd_handoff_pkg.tgz` 25KB)。

**若新 session 就在 88(PD 开发机)**:大部分已就位,只需 `git worktree add /home/docker/pd_dev_wt PD-dev` + 读本文。设计文档已抢救进 repo。

**若在全新机器(52 等),逐项补齐**:

1. **传包过去**(经 /home/docker 桥,和之前一样):宿主机 `scp /home/docker/pd_transfer/{PD-dev.bundle,pd_handoff_pkg.tgz} docker@<52>:/home/docker/pd_transfer/`。
2. **PD-dev 分支**(未推 origin,用 bundle):52 上 `cd /workspace/vime && git fetch /home/docker/pd_transfer/PD-dev.bundle PD-dev:PD-dev && git worktree add /home/docker/pd_dev_wt PD-dev`。
3. **文档**:`cd /workspace/vime && tar xzf /home/docker/pd_transfer/pd_handoff_pkg.tgz`(落 docs/design/ 四份)。
4. **mooncake ADXL build**(机器本地、必须):要么**重编**(源在 `/workspace/Mooncake`:`cmake .. -DUSE_ASCEND_DIRECT=ON -DUSE_CUDA=OFF -DPython_EXECUTABLE=/usr/local/bin/python3 && make -j install`),要么从 88 **拷 .so + python 包**(同 A2 架构):`/usr/local/lib/lib{mooncake_store,transfer_engine,mooncake_common}.so`+`ascend_transport.so` 和 `/usr/local/Ascend/cann-9.0.0/python/site-packages/mooncake/`。**别忘 LD_LIBRARY_PATH(§2)**。验证:`python -c "from mooncake import engine"` 不报错。
5. **vllm-ascend V1 连接器**(`/workspace/vllm-ascend` 是 overlay 非 NFS,52 需同版本):52 上先查 `grep -c "class MooncakeConnector" /workspace/vllm-ascend/.../kv_p2p/mooncake_connector.py`;缺则从 88 tar 整个 `/workspace/vllm-ascend` 同步(参照之前 vime 同步法)。
6. **polar**:52 是否需跑 polar 看部署;PD 单机是在 88 开发的(日志 remote_host=80.48.5.88),**确认 PD 到底在哪台机跑**——若仍在 88,52 只是备份,不必全套。

> ⚠️ **根因提醒**:设计文档、PD-dev 分支都曾**不在 origin/不在 git 树** → 换机器就丢。本次已把文档抢救进 repo、PD-dev 打了 bundle;**长久之计:新 session 上 PD-dev 后,把 docs/design 的 4 份文档 commit 进 PD-dev**,并考虑 `git push` PD-dev 到 origin,免得再靠 bundle。

> 源 memory(若新环境有则可 recall,没有也不影响——上面已内联):pd-mooncake-impl-status(最全逐提交)、mooncake-a2-transport-split、external-lb-dp-harmful-agentic-rollout、feature-eval-scaling-regime、follow-ref-before-diverging、project-repo-map、polar-host-control-hostctl、vime-validation-run-full-command。
