# PD 分离开发方案 — qwen3.6-35B-A3B(GDN-hybrid)· vime+polar · 单机 2P+2D

> 作者:Claude(ppo-adapt)· 2026-07-12 · 状态:设计(design-doc-first,未落码)
> 全部结论 **code-backed**(标 file:line);不看名字下结论。

## 0. TL;DR

- **PD 对我们这 GDN-hybrid 模型可行**,连接器侧 code-ready(`MooncakeHybridConnector` 真处理 Mamba/SSM 组)。
- **不做 KV 池化 #6**:它被 GDN×HMA 的 vLLM scheduler gap 卡死(`_update_requests_with_invalid_blocks` 单组假设,upstream RFC #36780 **closed not-planned**)。PD 用**不同连接器**,避开该 gap。
- **两个真依赖必须处理**:①**PD 编排层撞 #3**(vllm-router 丢 `return_token_ids`)——规避法=用可编辑的 **Python `load_balance_proxy`** 替代 vllm-router,自己加透传;②**mooncake 握手接线**(vime 现在只接 Nixl side-channel)。
- 工作量**有界**:vime PD 脚手架已存(server_groups / prefill+decode / bootstrap 端口),核心是"Nixl→mooncake-hybrid 换接线 + Python proxy + per-group kv_role/kv_port"。

---

## 1. 背景:为什么 PD 而不是池化

| 特性 | 连接器 | GDN 上的命运 |
|---|---|---|
| KV 池化 #6 | `AscendStoreConnector`(store) | ❌ 报 `invalid_block_ids` → vLLM scheduler `_update_requests_with_invalid_blocks`(`scheduler.py:2102` `# TODO(davidb): add support for hybrid memory allocator`)单组解包 `(req_block_ids,)=get_block_ids()` → 我们 HMA 2 组 → `ValueError: too many values to unpack`。upstream 无 fix(RFC #36780 closed not-planned)。 |
| **PD 分离 #5** | **`MooncakeHybridConnector`** | ✅ 传输失败走 `raise`(`mooncake_hybrid_connector.py:610`),**不喂 invalid_block_ids** → 不碰该 gap。 |

## 2. 已核实事实(code-backed)

### 2.1 连接器 `MooncakeHybridConnector` 真处理 GDN
文件:`vllm-ascend/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py`
- `class MooncakeConnector(KVConnectorBase_V1, SupportsHMA)`(:969),注册名 `MooncakeHybridConnector`(`kv_transfer/__init__.py:14-17`)。
- `_transfer_kv_cache_all_groups`(:545):`for i in range(self.hma_group_size)`(:576)遍历所有 HMA 组;**`if not isinstance(self.kv_cache_specs[i], MambaSpec)`(:581)** 显式区分 Mamba/SSM 组 vs full-attn 组分别处理。→ 实现了 RFC #36780 punt 的 SSM-FA hybrid 传输。
- `__init__`(:1069):`use_hybrid = not disable_HMA and any(非 FullAttentionSpec 组) and len(groups)>1`(:1108-1112)自动识别我们这种;`MambaSpec` → `need_truncate`(:1135/1145);逐组 `group_block_size`。
- 传输失败 `raise RuntimeError`(:610),**不产 `invalid_block_ids`** → 避开池化那个 scheduler gap。

### 2.2 连接器配置要求
- **`kv_port`**(`kv_transfer_config.kv_port`,:1092):握手基准端口,每引擎 `= kv_port + dp_rank×tp×pp`(:1091-1096)。**这是 mooncake 自有握手,不是 Nixl side-channel。**
- **`kv_role`**:P=`kv_producer`,D=`kv_consumer`(官方教程 `pd_disaggregation_mooncake_single_node.md`)。
- **HMA 必开**:`--no-vllm-disable-hybrid-kv-cache-manager`(已在 KV toggle)。
- **约束**:`pcp_size*dcp_size==1`(:1083,rollout 无 CP 满足);**P/D 的 TP 须一致**(`:1738` `assert prefill_tp_size==self.tp_size ... with Mamba`)。

### 2.3 vime PD 脚手架(已存在)
- `backends/vllm_utils/vllm_config.py`:`ServerGroupConfig(worker_type∈{regular,prefill,decode,placeholder,encoder}, num_gpus, num_gpus_per_engine, overrides)`;`ModelConfig.has_pd_disaggregation`(:103);`VllmConfig.from_prefill_num_servers`(:183:prefill_gpus=prefill_num_servers×num_gpus_per_engine,decode=total-prefill)。
- 拓扑经 **`--vllm-config` YAML**(server_groups)或 legacy `--prefill-num-servers`。
- `ray/rollout.py`:`ServerGroup`(:44,含 worker_type/router_ip/router_port);PD 每引擎分配 `disaggregation_bootstrap_port=get_port()`(:955)。
- **per-group `overrides`**(`_apply_vllm_overrides`,`vllm_engine.py:220`)可下发到各引擎——但 **跳过 `disaggregation*` 键**(:229-231);其余 `vllm_*` 键 setattr 到 args。→ 给 P/D 分设 kv-transfer-config 的入口(待确认 JSON 型 kv-transfer-config 能否走这条)。

### 2.4 vime PD 现状 = 纯 Nixl
- `vllm_engine.py:476-478`:prefill/decode worker 只设 `VLLM_NIXL_SIDE_CHANNEL_HOST/PORT = disaggregation_bootstrap_port`。**无任何 mooncake 痕迹**(grep 证实)。
- 而 **NixlConnector 的 hybrid SSM-FA 支持 = RFC #36780 closed not-planned** → vime 现成的 Nixl PD 对 GDN 不通。

### 2.5 router = vllm-router(= #3),PD 撞 #3
- vime router = `vllm_router`(`utils/http_utils.py:119` `from vllm_router.launch_router import launch_router`;`rollout.py:977 _start_router`)= **#3 那个 `vllm-router 0.1.14` 预编译 wheel,无源码、容器无 Rust 工具链**(`router_return_token_ids_passthrough.md §0`)。
- **#3 bug**:router 反序列化成 typed struct → 丢 `return_token_ids` → worker 回 `token_ids=null` → polar 轨迹 `response_ids=[]` vs `response_logprobs` 长度不符 → 训练饿死(该文档 §30)。
- **单引擎现在靠"polar 直连 worker `:15000` 绕过 router"跑**(§32);**多引擎(DP/PD)必须回编排层 → 撞 #3**(§34,§36 "DP 扩展前置项")。
- **PD 绕不开编排层**(P→D 两阶段,不能 polar 直连单 worker)→ **默认走 vllm-router = 撞 #3**。

---

## 3. 架构 / 目标拓扑(单机 4 卡推理预算)

```
polar gateway ──HTTP──> [PD 编排层] ──┬─> P engine(prefill, kv_producer, 卡4)
   (rollout 请求)                     └─> P engine(prefill, 卡5)
                                       ├─> D engine(decode,  kv_consumer, 卡6) ←─ mooncake ADXL pull KV
                                       └─> D engine(decode,  卡7)
actor(训练)= 卡 8-15(VIME_ROLLOUT_LOW_CARDS 同款低卡钉位,PD 下需重定拓扑)
```
- 2P + 2D,每引擎 1 卡(num_gpus_per_engine=1),TP1;或 1P+1D 各 2 卡 TP2(先冒烟)。
- **P/D TP 必须一致**(§2.2)。
- KV 流:D 收到 decode 请求 → 经 `kv_transfer_params` 知道对应 P → mooncake ADXL `batch_transfer_sync_read` 从 P 拉 KV(含 Mamba state)。

---

## 4. 开发工作项(WBS)

### W1 — 连接器选型:P/D 用 MooncakeHybridConnector(不是 Nixl)
- 经 per-group `overrides` / `--vllm-kv-transfer-config` 给:
  - P group:`{"kv_connector":"MooncakeHybridConnector","kv_role":"kv_producer","kv_port":<P_port>, ...}`
  - D group:`{...,"kv_role":"kv_consumer","kv_port":<D_port>, ...}`
- **待确认**:`_apply_vllm_overrides` 跳过 `disaggregation*` 但 kv-transfer-config 不带该前缀,应能走 `vllm_kv_transfer_config` setattr 路径;需实测 JSON 值下发无损。

### W2 — 握手接线:mooncake kv_port 替 Nixl side-channel
- 改 `vllm_engine.py:476-478` prefill/decode 分支:当连接器=mooncake 时,把 `disaggregation_bootstrap_port` 映射成连接器的 `kv_port`(塞进 kv-transfer-config),**而非** `VLLM_NIXL_SIDE_CHANNEL_PORT`。
- 加一个 `disaggregation_backend ∈ {nixl, mooncake}` 开关(retro-compat:默认 nixl 不动老行为)。

### W3 — PD 编排层 + #3 规避(关键决策)
- **决策:用教程的 Python `load_balance_proxy_server_example.py` 替代 vllm-router 做 PD 编排**(vllm-ascend examples 里,纯 Python、可改)。
  - 它做 prefill→P / decode→D + 传 `kv_transfer_params`。
  - **在这个 Python proxy 里加 `return_token_ids` 透传(几行,Python)→ 绕开 #3**(#3 卡在 Rust vllm-router 无源码;Python proxy 我们全控)。
- polar 端点指向该 proxy(替代直连 worker:15000)。
- 备选(不推荐):等 #3(取 vllm-router 0.1.14 源码 + Rust 建 + 换 wheel,需用户/构建环境 + 动 polar)。

### W4 — RL 专属
- **权重同步**:actor(update_weights=true)在 PD 下 P+D 引擎都要收权重回灌(vime sleep-mode wake_up)。需验 weight sync 覆盖两组引擎。
- **prefix-cache**:教程 PD `--no-enable-prefix-caching`;我们全栈开着 prefix-cache。PD 下 P 做 prefill(prefix-cache 对 P 有益)、D 只 decode。需定 P 开/D 关,并验与 mooncake KV 传输不冲突。
- **拓扑**:`VIME_ROLLOUT_LOW_CARDS` 是"rollout 占低卡"逻辑;PD 把 rollout 拆成 P+D 两组,卡位分配要重定(placement_group.py 可能要扩)。

### W5 — per-group 配置下发验证
- 确认 `_apply_vllm_overrides` 能把每组不同的 kv-transfer-config(不同 kv_role/kv_port)正确下发到对应引擎进程。

---

## 5. 开放风险 / 必须实跑核实

1. **Mamba/SSM state 的 P→D 传输正确性**:连接器有代码(MambaSpec 分支),但**未 e2e 验**——SSM 是定长 recurrent state,传错=D 侧生成错但不崩,要靠 token-faith 抓。**最高优先级验证点。**
2. **W1 kv-transfer-config 按组下发**是否无损(JSON 值 + `disaggregation*` skip 逻辑的边界)。
3. **prefix-cache × PD × mooncake** 三者共存(P 开 prefix-cache 时,KV 块语义是否影响 mooncake 传输)。
4. **#3 规避的 Python proxy** 是否支持我们要的 `/inference/v1/generate`(vime rollout 用的端点,`vllm_rollout.py:82`)——教程 proxy 是 `/v1/chat/completions`,可能要适配 vime 的 generate 端点。

---

## 6. 验证计划

**权威闸 = token-faith(TIS≈1),一步到位、优先看。** 理由:TIS = exp(logprob_train − logprob_rollout),训练侧用**正确 KV 全重算** logprob 与 rollout 侧比 = off-policy 校正本身。PD 传错离谱 → 垃圾 token 在错状态高概率、正确模型低概率 → TIS 远离 1(clipfrac/kl 同炸),**必抓**;传错轻微 → TIS≈1,而此时**训练本就未受损**(IS 权重≈1)。故 **TIS≈1 ⟺ rollout 对训练保真 ⟺ 训练正确**,正是我们的判据。
- 全栈 config(`VIME_ROLLOUT_LOW_CARDS`/2×4/chunk-lmhead/mtpg32768,见 [[vime-validation-run-full-command]]),**step0+step1 TIS≈1 + logprob_abs_diff≲0.05**。
- nuance(不影响):TIS 保证"训练保真",非"PD 输出与非-PD 逐字一致";而逐字一致(推理确定性)**不是我们的判据**,RL 要的是保真 logprob。

**可选的便宜预冒烟(非正确性要求,只为省时间早抓 gross 崩坏)**:
- **差分贪婪**:同 prompt `temperature=0`,单引擎 vs PD 逐字比。纯推理 ~2min(TIS 要跑完整 rollout+train ~15min)→ PD 若 gross 崩坏(D 吐垃圾)这里先炸,省得白等一轮训练。通过不代表对(仍以 TIS 为准),不通过=肯定错。
- 引擎级:`batch_transfer_sync_read` ret≥0 + 无 raise + proxy 日志 P→D 路由 + `kv_transfer_params` 传递(master 不需要,P2P 非 store)。

**分阶段**:先 **1P1D**(各 2 卡 TP2)过引擎级 + 差分贪婴预冒烟 → 再 **2P+2D**(各 1 卡)跑 **token-faith 权威闸**。

---

## 7. 回滚 / 安全

- 全部改动隔离在 vime fork + 一个 Python proxy;**vllm-router / polar 不动**(用 Python proxy 旁路,而非改 router)。
- 失败回退:polar 端点切回单引擎直连 worker:15000(现状),PD 分支 env 关。
- `disaggregation_backend` 默认 nixl → 非 PD-mooncake 场景零回归。

---

## 8. 依赖关系小结

| 依赖 | 状态 | 本方案处理 |
|---|---|---|
| mooncake 库(ADXL build) | ✅ 已编(v0.3.9) | 复用 |
| MooncakeHybridConnector GDN 支持 | ✅ code-ready | 复用,实跑验 Mamba state |
| vime PD 脚手架 | ✅ 已存(Nixl 接线) | 改接 mooncake(W2) |
| #3 return_token_ids router 透传 | ❌ 硬阻塞(Rust 无源码) | **Python proxy 旁路规避**(W3) |
| router(vllm-router) | 不改 | 用 Python proxy 替代做 PD 编排 |

**结论**:PD 可推进,不必等 #3(用可编辑 Python proxy 旁路)。核心工作 = W1-W3(连接器+握手+proxy),最大不确定 = Mamba state P→D 传输实跑正确性(§5.1)。

---

## 9. 实现状态(2026-07-12 · worktree `/home/docker/pd_mooncake_wt` @ feature/pd-mooncake)

**W1-W4 全部落码 + committed**(隔离 worktree,live `/workspace/vime` 未碰,py_compile 全过):
- **W1+W2** `4512088f`:vime `--disaggregation-backend {nixl,mooncake}`(默认 nixl 零回归)+ `build_vllm_cmd_and_env` 按 worker_type 注入 `--kv-transfer-config`(MooncakeHybridConnector,P=kv_producer/D=kv_consumer,kv_port=bootstrap_port),取代 Nixl side-channel。
- **W3** `0b999fa3`:`scripts/pd_mooncake_proxy_server.py`(拷 vllm-ascend `load_balance_proxy` + 加 `/inference/v1/generate` 路由 + `build_prefill_request` 适配 `sampling_params.max_tokens`;GenerateResponse 主路径原样透传)。
- **W4** `a8fea1d5`:`rollout.py` static-PD 路径 backend=mooncake 时 `_start_mooncake_pd_proxy` 起 proxy(解析 P/D `http://host:port` → `--prefiller-hosts/ports`+`--decoder-hosts/ports`,bind router_port),替 vllm-router 避 #3。

**还差(才能跑)**:
1. **启动脚本** `run-qwen36-35b-polar-minimal.sh` 传 `--vllm-config <2P2D YAML>` + `--disaggregation-backend mooncake` 给 `train_async.py`(脚本目前无 PD hook,小改;或 env 覆盖)。
2. **polar** `profile.vime.yaml` 的 `sglang_router_url` → proxy(`router_port`),非 worker `:15000`。
3. **验证**:1P1D 差分贪婪(Mamba-state 正确性)→ 2P+2D token-faith。**要卡**,等空。

**2P+2D vllm-config YAML**(卡 4-7:2P 4-5 / 2D 6-7,各 TP1;或 1P1D 各 TP2 先冒烟):
```yaml
vllm:
  - name: actor
    update_weights: true
    server_groups:
      - {worker_type: prefill, num_gpus: 2, num_gpus_per_engine: 1}
      - {worker_type: decode,  num_gpus: 2, num_gpus_per_engine: 1}
```

关联:[[mooncake-a2-transport-split]] [[router_return_token_ids_passthrough]] [[vime-validation-run-full-command]] [[follow-ref-before-diverging]]
