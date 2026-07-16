# PPO 单机功能测试 · 工作日志(自主 /loop)

> 目标:单机小 batch 打通 vime PPO(actor+critic 共卡 offload 时分)训练功能。
> 起始:2026-07-15。执行模式:自主 /loop,遇 bug 最小改动修复、全程记录、语义 commit。

## 用户决策(2026-07-15 离开前拍板)
1. **Commit 范围**:整棵脏树按语义全提交(不只本次改动)。
2. **Polar**:若不在 → hostctl 自动重启(restart_polar_stack,保 :8001,绝不 kill)。
3. **修复边界**:允许必要时做较大改动(在日志记录清楚的前提下)。
4. **验收标准**:首步 PPO e2e 通过(F-PPO-1/3、critic value [T,1]、policy/value 双 loss 有限、TIS≈1)+ ~3 步跨权重更新稳定 → 停并汇报。

## 就绪前置(本 session 已完成)
- A2:PPO 脚本 `scripts/run-qwen36-35b-polar-ppo.sh` 从当前 GRPO 重 fork,diff 只剩 4 处 delta(header/RUN_ID/PPO_ARGS/调用处),含 OOM 修复 FEAT_TRAIN_EXPANDABLE。bash -n ✓。
- F-PPO-1 gate:`chunked_lm_head_patch.py` 补 value-head 例外(shape[0]==1 不旁路),py_compile ✓。
- 机制确认:critic/actor 共卡(同 PG)+ offload 时分(train 进 onload/出 offload)+ value 依赖串行化;每步 HCCL churn(destroy/reload PG ×3 轮 + rollout 组断/连),F-PPO-3 最高风险 regime。

## 测试计划
- **Run-A**(NUM_ROLLOUT=1,单步破冰):position 路径(不设 RESOURCE_LAYOUT),小 batch(RB=2/N=2/GBS=4),QWEN36_CHUNK_LMHEAD=1,OOM 三件套开。验:offload/wake+HCCL churn 穿过不崩/不卡、critic value [T,1]、双 loss 有限、TIS≈1。
- **Run-B**(NUM_ROLLOUT=3~4):跨权重更新稳定,无 OOM、无 HCCL 累积失败。
- 命令基线见下方运行记录。

## 运行记录
<!-- 按时间追加:每次动作、观察、bug、修复、commit -->

### [启动] 2026-07-15 — 迭代 1
- 建本日志。git=feature/lb-proxy,17 脏改动。
- **脏树按语义 4 提交**(工作树已干净):
  - `05a64456` feat(dp): external-LB DP + 跨DP EP + 单测 + 设计文档
  - `e48a4aa1` fix(oom): reserved ratchet 冻死修复(每步 empty_cache)+ NPU 显存探针
  - `9496e427` chore(scripts): polar-minimal 特性开关 + start.sh
  - `6102a33b` feat(ppo): 单机 PPO 适配(F-PPO-1 gate + 重 fork PPO 脚本 + 本日志)
- polar :8080 LISTEN ✓(无需 hostctl 重启);hostctl 通道 `/home/docker/polar_e2e/` 备用。
- 卡 4-15 空闲(GRPO 已停)。

### [Run-A] 2026-07-15 — PPO 冒烟(NUM_ROLLOUT=4,单机 position 路径)
- RUN_ID=`qwen36_polar_ppo_smoke1`,log=`/home/docker/logs/train_qwen36_polar_ppo_smoke1.log`
- 命令(后台):
  ```
  CURRENT_IP=80.48.5.88 MASTER_ADDR=80.48.5.88 NNODES=1 NPUS_PER_NODE=12 \
  RUN_ID=qwen36_polar_ppo_smoke1 QWEN36_CHUNK_LMHEAD=1 \
  FEAT_TRAIN_EXPANDABLE=1 VIME_EMPTY_CACHE_PER_STEP=1 VIME_MEM_PROBE=1 FEAT_LB_PROXY=1 \
  NUM_CRITIC_ONLY_STEPS=0 ROLLOUT_BATCH_SIZE=2 N_SAMPLES_PER_PROMPT=2 GLOBAL_BATCH_SIZE=4 \
  NUM_ROLLOUT=4 ROLLOUT_MAX_RESPONSE_LEN=8192 MAX_TOKENS_PER_GPU=32768 \
  SAVE=/workspace/Qwen3.6-35B-A3B_vime_ppo_smoke bash scripts/run-qwen36-35b-polar-ppo.sh
  ```
- 冒烟提速:response_len 降到 8192(功能验证不需 32k);其余对齐生产(chunk-lmhead/OOM 三件套/lb-proxy)。
- 盯:①model init 不崩(F-PPO-1 gate 生效)②offload/wake + HCCL churn 穿过不崩/不卡(F-PPO-3)③critic value [T,1] ④policy/value 双 loss 有限 ⑤跨 3-4 步稳定 + 无 OOM。
- **[17:30 观察 · init 通过]**
  - 全 rank post-init:expandable=2/2、exp_reserved=27.99(OOM 修复携带 OK);model init **没崩**(F-PPO-1 gate 未破坏 init)。
  - ✅ **critic value head 正确**:`model.py:187 reinitialize critic output_layer.weight checkpoint=(248320,2048) runtime=(1,2048)`([repeated 7x]=8 critic rank)→ critic backbone 从 ref_load 载入 + 标量头 (1,2048) + reinit 生效(A4 + model_provider 链路验证)。
  - vLLM rollout 引擎起(/update_weights route、V1 engine v0.21.0、TP=4);无 Traceback/EI00/OOM。
  - 待验:初始权重同步(首次 HCCL churn)→ rollout perf → 首步 [MEM-EMPTY](F-PPO-1 真实验证点 get_values [T,1] + offload/wake churn + 双 loss)。
- **[并行预读 · 首步风险路径 map]**(趁 rollout 慢,提前读好便于 Monitor 一响即时判读)
  - **F-PPO-1 断言点** = `loss.py:545 assert logits_chunk.size(-1)==1`(在 `get_values`)。gate 失效→critic 拿 hidden `[R,2048]`→此处崩(Monitor `size(-1)` 抓)。首步 get_values 不崩 = F-PPO-1 通过。
  - **critic reinit 正确**:`model.py:200 weight.data.normal_(0,0.02)+bias.zero_()`。
  - **F-PPO-3 进程组重建(D1)基本 de-risk**:`monkey_patch_torch_dist`(actor.py:64)早于 model init(:136)装;Megatron(parallel_state.py:236)+ MindSpeed CP ulysses/ring(model_parallel_utils.py:52/102/113/157/160)全走 `torch.distributed.new_group`→全被 ReloadableProcessGroup 包→offload/wake 全可 destroy+reload。**残留低风险**:reload 只还原 ranks+backend、不还原 `pg_options`(nccl_comm_cfgs)→ CP ring 重建成默认 options。若首次 offload/wake 崩,查 pg_options / 绕过 new_group 的组,而非机制缺失。
  - DP 提交语法复验:rollout/arguments/vllm_engine/update_weight py_compile ✓。
- **[18:35 · 🔴 BUG 确认 · F-PPO-3 offload 权重同步死锁 vLLM 引擎]**
  - **现象**:67min 一个 rollout 都没 accept;**7395 个 polar session 全 `traces=0`**(no usable trace→dummy);vLLM **0 个生成请求**(`POST /v1/chat/completions`=0)。
  - **定位**:直接打 :8001 与直连引擎 :15000 **都 504 Gateway Timeout**;vLLM 引擎进程活着但**最后一条日志停在 17:35:14**(初始 update_weights 那刻),之后 1 小时零日志零响应 → **引擎在初始 offload 权重同步时死锁冻住**。
  - **PPO 特有(非 polar/环境)**:同 polar 上最近 3 个 GRPO run(092134/144301/150552)全正常(vLLM 收 11389/458/1812 真实请求、traces=0≈0)。排除"权重加载失败"红鲱鱼(GRPO 有 13562 个同样良性 warning 仍跑到 perf21)。
  - **根因方向**:PPO 的 `update_weights` 走 `reconnect_rollout_engines=True`(offload+use_critic+!colocate)→ `wake→connect_rollout_engines→broadcast→sleep(disconnect_rollout_engines+destroy_process_groups)`;疑似 sleep 的 disconnect/destroy 把 vLLM 引擎留在权重广播的 HCCL collective 挂起态(或广播 weight 数不匹配 vLLM 等待)。GRPO 无 offload/不 disconnect,故引擎一直响应。**这正是 doc 标注的 F-PPO-3 "最高风险·从未验证" regime。**
  - **动作**:停死锁 run(ray stop,polar 未动);调查 weight-sync connect/disconnect/broadcast 代码找最小 fix。**注**:冒烟设的 ROLLOUT_MAX_RESPONSE_LEN=8192 非本 bug 因(截断仅 1 次)。
- **[19:05 · 诊断精修(纠错)]**
  - **纠正**:actor 侧 `disconnect_rollout_engines_from_distributed`(update_weight_from_distributed.py:459)是 **no-op**(docstring:trainer 侧 comm 故意不拆避 CUDA-graph self-deadlock、engine 侧 destroy 也 no-op)→ **"disconnect 拆组挂死"假设作废**。
  - **精确定位**:直连 :15000 也 504 → 排除 polar/version-span,是 **vLLM EngineCore 自身死锁**。时序:收权重(17:35:14)→ update_weights 完成 197.9s(POST /update_weights 返回=engine handler 完成)→ resume(continue_generation)→ **之后在生成时 EngineCore 卡死**。
  - **已排除**:disconnect(no-op)/polar+version-span(直连也 504)/8192 截断(仅1)/abort(仅1)/权重加载 warning(GRPO 13562 个仍正常)。
  - **唯一 PPO 差异 = offload**(actor wake/sleep 环绕 update_weights);但每个具体假设 GRPO 都相同却正常 → 确切失效点静态看不出。
  - **下一步 = py-spy 挂死引擎**:死锁在**初始 update_weights**(launch 后 ~11min,早于长 rollout),复现快。重启 smoke2 → 等初始 update_weights 后引擎 504 → py-spy EngineCore + actor rank0 拿确切死锁栈 → 定点 fix 或带栈升级给用户。
- **[19:15 · py-spy 反转 + 发现我的 confound]**
  - **py-spy 挂死引擎(2309736,post-kill 仍活)**:主线程栈 = `run_busy_loop→_process_input_queue→queue.get()` → **引擎 IDLE 等请求,不是 HCCL 死锁**;TP worker(2309781-84)全在 `do_poll`(等消息,非 NPU ioctl)。→ 真相是**请求根本没到引擎**,不是"引擎在 collective 挂死"。之前"EngineCore 死锁"判断**过头了**。
  - **🔑 我的 confound**:smoke1 **丢了工作正常的 GRPO run(092134)的一堆 vLLM FEAT 开关**——GRPO 全开 `FEAT_FLASHCOMM1/PREFIX_CACHE/MULTISTREAM_SHARED_EXPERT/STATIC_KERNEL/HCCL_AIV`,我 smoke1 一个没设,还把 response_len 降 8192。→ **"深锁"很可能是我配置发散(掉了 GRPO 必需的 vLLM 特性/additional-config)导致引擎路由异常,未必 F-PPO-3。**
  - **动作**:smoke2 = **完全对齐 GRPO 092134 配置(全 FEAT 开关 + response_len 默认)+ 只加 PPO args**,隔离 confound。若仍卡在 update_weights → 真 PPO/offload,再 py-spy(pre-kill)拿真栈;若通 → 我的配置发散是因。**教训:测新特性务必只改一个变量,别顺手动一堆配置(呼应 [[vime-validation-run-full-command]])。**
- **[19:25 · 🔴 smoke2 撞 F-PPO-2 EI0013(跨域 RoCE)· 单机 PPO 真 blocker 定性]**
  - smoke2(对齐 GRPO 全配置+PPO)**6min 撞 EI0013**:rank3 device **7↔11** 跨 HCCS 域 RoCE CQE 错误 → HCCL watchdog terminated → 崩。
  - **拓扑实证**(npu-smi topo):HCCS 两域 = **0-7** 和 **8-15**(域间 PIX/PHB/SYS 非 HCCS→RoCE)。position 路径 actor 落 **4-11 横跨两域** → 训练 collective(rank3=card7 ↔ rank7=card11)跨域 → EI0013。
  - **两 smoke 失败统一定性**:smoke1(无 HCCL_AIV)引擎无请求 + smoke2(有 HCCL_AIV)EI0013 —— **都是"单机 position 路径 PPO"的 doc 预测 blocker(F-PPO-2 跨域/F-PPO-3 offload)**;HCCL_AIV × 跨域是 EI0013 触发器。**工作正常的 GRPO 用 explicit layout 钉卡不跨域,我用 position 路径才跨域。**
  - **✅ 修复 = route-b(doc 明定,前置就绪)**:
    - `resource_layout_actor_domain2.yaml` 已存在(**actor 钉 8-15 同域→免 EI0013**,rollout 4-7,polar 0-3,专为此建)。
    - F-PPO-2 Lock fix **已在 LIVE**(rollout.py:395 num_cpus=0)。
    - A1(layout 加 critic placement + 删 "does not support critic" raise + A3 critic save 分目录)= `/home/docker/ppo_adapt_dev/A1-A3_placement_critic.patch`(非 git 格式,手工应用 placement_group.py:171/180/276)。
  - **下一步**:应用 A1 → 跑 smoke3(RESOURCE_LAYOUT=resource_layout_actor_domain2.yaml,对齐 GRPO 配置+PPO)。避 EI0013 后若仍卡 → 才是 F-PPO-3 offload 真问题,再 py-spy。**注:route-b 是 doc 标注"需用户协调"的深水项;已授权"较大改动",继续推进。**
- **[19:41 · ✅ route-b 生效 · F-PPO-2 已解]**(smoke3)
  - **actor 真钉 8-15 同 HCCS 域**、rollout 4-7(placement_group.py:133 逐 bundle 证实);A1 layout critic placement 生效、无 "does not support critic" raise。
  - **dist-init WORLD_SIZE=8(8/8)** —— F-PPO-2 step1 Lock fix(num_cpus=0)生效,无 7/8 抢占。
  - **EI0013=0**(smoke2 同期 6min 已崩;smoke3 过关)—— 同域训练 collective 避开跨域 RoCE。critic value reinit 触发。
  - **待验(下一判别点 update_weights ~19:47)**:weight-sync=actor 8-15↔rollout 4-7 跨域(会否 EI0013?)+ F-PPO-3 offload(会否 smoke1 那种引擎不服务?)。注:vLLM worker 19:37 有 decorators.py:321 WARNING traceback(疑良性,待观察)。
- **[19:47 · 🎯 F-PPO-3 根因完全定位 + 单机 PPO 功能测试总结]**
  - **smoke3 判别铁定**:route-b(干净 layout,EI0013 已解、dist-init 8/8)**仍 148+ traces=0、504** → "引擎不服务"是真 F-PPO-3,非 position/config confound。
  - **py-spy(pre-kill,干净)**:EngineCore + TP workers 全 **IDLE**(EngineCore `queue.get`、worker `shm_broadcast dequeue`),**非 HCCL 死锁**;API server 在 uvloop 事件循环。→ 引擎健康但收不到请求。
  - **根因铁证**:`/pause`=1 但 `/resume`=**0**(GRPO 是 46/44 配对);update_weights 窗口有 **`ray.exceptions.ActorUnavailableError: RpcError: Socket closed (rpc_code 14)`**。→ **offload 权重同步期间 rollout 引擎 Ray actor RPC socket 断了 → 打断 pause→resume → 引擎永久停在 abort-pause 态 → 504 → 全 traces=0 → rollout 攒不出 → 卡死。** GRPO 无 offload、RPC 稳定,故正常。
  - **修复方向(深水,待用户拍板)**:①resume(continue_generation)放 finally/加重试,pause 后必 resume;②查 offload 的 reconnect_rollout_engines(每 update_weights connect/disconnect HCCL 组)为何断 rollout 引擎的 Ray RPC;③abort→keep 模式(但那是 option-3 设计)。

  ### ★ 单机 PPO 功能测试总结(截至 2026-07-15 19:47)
  | 项 | 结论 |
  |---|---|
  | **F-PPO-1** chunk-lmhead 掐 critic value head | ✅ gate 生效(init 不崩、critic value head reinit [1,2048] 正确) |
  | **F-PPO-2** 单机跨域 EI0013 | ✅ **route-b 解决**:A1(layout critic 共卡,已应用+提交)+ `resource_layout_actor_domain2.yaml`(actor 钉 8-15 同域)→ dist-init 8/8、EI0013=0 |
  | **F-PPO-3** offload 权重同步后引擎不服务 | 🔴 **真 blocker,根因=pause 无 resume(ActorUnavailableError 断 RPC)**,已精确定位,修复待拍板 |
  | OOM 修复 | ✅ 携带正常(smoke 全程 expandable 2/2) |
  - **净结果**:PPO 端到端在单机跑通了 init/critic/EI0013,**唯一剩 F-PPO-3(offload×rollout-RPC)一个精确 blocker**。三 smoke run 全停,卡空,polar 未动。

- **[2026-07-16 02:00 · ⚠️ smoke4 加 [WSYNC-DBG] 日志——推翻上面的 F-PPO-3 根因,上表 F-PPO-3 行作废]**
  - 给 pause/resume 加日志(`update_weight_from_distributed.py` + `vllm_engine.py`)后 smoke4 实测:`PAUSE done returns=[Response 200]` + `post-broadcast barrier rank=0`(rank 没变)+ `RESUME block 进了` + `RESUME done returns=[Response 200]` → **pause 和 resume 都 200 OK 正常执行**。复查 smoke3 `/resume`=**1**(早先 `POST /resume` grep 漏了)。`ActorUnavailableError` 栈在 `train_async.py:43 ray.get(rollout_data_future)`,**不在 resume 路径**。
  - **→ "pause 无 resume 卡死"是错的(grep 假象)。教训:别用弱 grep 下强根因,要运行时日志。**
  - **F-PPO-3 真相(两个 offload bug,均未 root-cause)**:①**引擎 pause(abort)+resume 都 200 后仍不服务**(:8001/:15000 都 504,请求到不了 EngineCore,py-spy 证引擎 IDLE);②**训练侧崩** `Megatron param_and_grad_buffer.py:908 reset(): tensor data not allocated` + Ascend `ERR01003`(offload 释放 grad buffer 后 reset 拿到未分配 tensor)。
  - **下一步**:分开查 (a) vLLM abort-pause+resume 后 API server→EngineCore 为何不通;(b) offload grad buffer 生命周期(NPUWeightOffloader 漏 onload grad buffer?)。别急下单一根因。smoke4 已崩停,卡空 polar 未动。

- **[2026-07-16 03:15 · ✅ 走 Option A:NPU offload 对齐 slime,用 torch_memory_saver 换掉手搓 NPUWeightOffloader — F-PPO-3 bug② 从根上消除]**
  - **溯源 NPUWeightOffloader**:非 vime/slime 上游,是同事 ZhihaoSun `75293035 "tmp for gogogo"`(2026-06-26)手搓的 NPU 替代品。slime 官方(含 slime-ascend)PPO 路径**统一用 torch_memory_saver**(actor.py:180/189 无条件 pause/resume,NPU 也走),**无此 bug**。vime 本就继承了 slime 的 torch_memory_saver + PPO 支持,同事只是**额外**加了 NPU 分支替代品。
  - **bug② 根因**:`NPUWeightOffloader._release_ddp_buffers` 把 `grad_data` storage `resize_(0)` + `p.main_grad=None`,但 `onload()` **只还 param.data、从不还 grad buffer** → 下个训练 step `param_and_grad_buffer.reset()` 撞未分配 tensor(`ERR01003`)。Python 层 storage resize **本质上无法 transparent**(resize 回来是新分配,main_grad view 全悬空)。
  - **验证优先(用户要求"先实测再改")**:
    - **差点误判**:首测报 `LD_PRELOAD cannot be preloaded` → 差点断言"torch_memory_saver 昇腾不可用、同事手搓有理"。**实为我路径 bug**——.so 在 `site-packages/` 顶层,我多套了一层 `torch_memory_saver/` 子目录。slime 用 `dirname(dirname(__file__))` 两层才对。
    - **正确路径重测(card 0):`.so 的 NEEDED 含 `libascendcl.so`** = 昇腾专用 build,hook `aclrtMalloc`;`with region(enable_cpu_backup=True)` 内分配 4.3G → `pause()` 释放 4.3G HBM(60.73→65.03G)→ `resume()` **数据完整**(x.sum 1073741824→1073741824,match=True)。**torch_memory_saver 在本 A2 机确实能 offload/还原**,allocator 层做、不破坏 grad_data view → 无 bug②。
  - **改动(4 处,全镜像 slime,commit 见下)**:
    1. `actor.py` imports:`from torch_memory_saver import torch_memory_saver`(无条件,删 NPUWeightOffloader import + is_npu gate)。
    2. `actor.py` offloader-init:删 NPUWeightOffloader/storage_resize_hook 分支,统一 slime 式 `memory_margin_bytes` 设置。
    3. `actor.py` sleep/wake:`torch_memory_saver.pause()/resume()` 无条件(对齐 slime:180/189)。
    4. `actor_group.py`:offload_train 块拆 `and not is_npu()`(NPU 也挂 LD_PRELOAD hook + TMS_INIT_ENABLE + CPU_BACKUP)。
    - NPUWeightOffloader/storage_resize_hook 现已 **orphaned 无人 import**;`npu_weight_offloader.py` 保留(死代码,低风险不删),已清我加的 ② 探针。
  - **下一步 = smoke5**:GRPO 对齐配置(FLASHCOMM1/PREFIX_CACHE/MULTISTREAM/STATIC_KERNEL,不带 HCCL_AIV,domain2 layout)冲 update_weights→真生成→[MEM-EMPTY]→首步+3步。重点看:bug② 是否消失(offload/onload 不再崩 reset);bug①(引擎 pause+resume 后不服务)是否随 offloader 换掉一并消失(① 探针在 vllm scheduler set_pause_state 仍在,观察 pause/resume 时序+有无 PAUSED_NEW 饿死)。
