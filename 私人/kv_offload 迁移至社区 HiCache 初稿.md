## 1. 结论

当前的 `kv_offload` 是一种主动的 KV 管理策略，和社区 HiCache 的按需回灌路径不同。
迁移到社区的建议是新增开关分支，复用 `HiCacheController` 和 host KV pool，将 prompt KV 备份到 host，并在 `attention_backend=nsa` 的 decode 路径按 `Indexer` 的 topk miss 从 host 回填到 GPU 的稀疏 KV 缓冲区（对齐 `sparse_kv_buffer` 语义）。
其他场景保持社区原逻辑，并且尽量不新增或修改算子。

### 目标

- 在社区 HiCache 框架内支持 `enable_kv_offload` 行为
- 默认不改变社区行为，新增逻辑仅在开关开启后生效
- 以增量开发方式推进，优先复用社区已有模块与接口
- 非必要不引入定制算子，优先调用已有算子

### 迁移前提

- 目标场景是 `attention_backend=nsa` 的 decode 路径
- decode 侧需要可用的 host KV pool，KV-only 模式下对 NSA 选择只承载 `kv_cache` 的 host pool，例如 `MLATokenToKVPoolHost`
- host KV pool 选用 `hicache_mem_layout=layer_first`，KV 可按 indices 零散搬运；host pool 的 `alloc` 仍要求 size 按 `page_size` 对齐

### 取舍

- 不引入 L3 storage，比如 `hicache_storage_backend`，从而避免 `page-first` 或者 `page-first-direct` 存储布局
- 不迁移 `enable_doul_attn` 类的 hit/miss 分流，统一要求 attention 在读取 KV 前等待 cache load 完成
- 不大幅改动下，无法做到 `Prefill` 节点的 L1 的 cache 直接写入到 `Decode`节点的 L2 上，只能通过先拿着完整的 cache 先计算一层的 Attention，同时异步搬运到 L2 上，从而尽量掩盖这一部分的开销 

## 2. 当前实现逻辑

**Offload 数据范围（`attention_backend=nsa`）**

- 仅 offload `kv_cache`，每 token 每 layer 656B，对应 `NSATokenToKVPool.kv_cache_dim=656`
- 不 offload `indexer_buffer`，每 token 每 layer 132B，对应 `NSATokenToKVPool.index_k_with_scale_buffer`
- 结果是 decode 侧 `Indexer` 不需要等待 host load，即可基于 GPU 常驻的 index 做 topk

**Prefill 侧**

- `scheduler.py` prefill 结束后调用 `send_kv_cache`
- 发送接口是 `MooncakeKVTransferAgent.send_kv_caches(...)`
- 发送的 `kv_indices` 来自 `req_to_token_pool.req_to_token[req_pool_idx, :prompt_len]`
- 开 `enable_kv_offload` 时，会额外传 `select_sparse_kv_indices`，默认取 prompt 的前 `buffer_size` 个位置
- 发送后会走 `tree_cache.cache_finished_req(req)`，把 prefill 侧的 KV 释放掉

**Decode 侧**

- `scheduler.py` 里用 `recv_kv_caches(loc=out_cache_loc, ...)` 把 prompt KV 收到 decode 侧
- 收完以后用 `UpdateReqToTokenPool` 把第一段 prompt 的 CPU slot 和 GPU sparse slot 对上，写进
  - `cpu_slot_to_gpu_slot`
  - `cache_ids/cache_ids_inverse/cache_lens`
  - `out_cache_loc_sparse`

**Decode 每步每层**

- `deepseek_v2.py` 里先跑 `indexer` 得到 `topk_indices`
- `PrepareIntraLayerH2D` 或 `PrepareIntraLayerH2DNoLRU` 根据 `cpu_slot_to_gpu_slot` 找 miss，并给出 `evict_ids`
- `IntraLayerH2D` 在 `trans_stream` 把 miss 的 KV 从 CPU `kv_buffer` 拷到 GPU `sparse_kv_buffer`
- Attention 侧通过 `OffloadArgs` 里的 `cuda.Event` 等待拷贝完成

### LRU 与后处理

- LRU 目标：用固定大小的 GPU `sparse_kv_buffer` 当 L2，只保留每层当前 `topk` 需要的 token KV，其他 KV 常驻在 CPU `kv_buffer`
- LRU 状态都在 GPU `ReqToTokenPool`：
  - `cpu_slot_to_gpu_slot`：CPU slot 到 GPU sparse slot 的映射，值为 `-1` 表示不在 GPU
  - `cache_ids/cache_ids_inverse/cache_lens`：每个请求的 LRU 列表与反查表
  - `out_cache_loc_sparse`：该请求可用的 GPU sparse slot 列表，用于给 miss 分配位置
- 每层更新：
  - `PrepareIntraLayerH2D` 基于 `topk_indices` 计算 miss，并在 GPU 上更新 LRU，产出 `evict_ids` 与新的 slot 分配
  - `IntraLayerH2D` 只搬 miss token 的 `kv_cache`，写入对应的 GPU sparse slot
- 每 step 后处理：
  - `CompactCacheIdsBlock` 压缩 `cache_ids` 的空洞并重建 `cache_ids_inverse`
  - `ReorderOutCacheLocSparse` 重排 `out_cache_loc_sparse`，保证前部是当前有效 cache 对应的 GPU slot

## 3. 社区实现逻辑

**Prefix 匹配和回灌**

- `schedule_policy.py` 调 `tree_cache.match_prefix(...)`，会拿到 `host_hit_length` 和 `last_host_node`
- `host_hit_length > 0` 时会调用 `tree_cache.init_load_back(last_host_node, host_hit_length)`
- `hiradix_cache.py` 的 `load_back` 会把一整段 `host_value` 搬回 GPU
- `len(host_indices) < load_back_threshold` 会直接不搬

**PD disaggregation 的 KV 发送和接收**

- Prefill 侧在 `disaggregation/prefill.py` 调 `KVSender.send(page_indices, state_indices)`
- 对 `state_type=nsa`，`state_indices` 会额外包含 `index_k_with_scale_buffer`，保证 decode 侧有可用的 `indexer_buffer`
- Decode 侧先在 GPU 分配 `kv_loc`，再 `KVReceiver.init(page_indices, ...)`
- Mooncake backend 的实现是 decode 把自己 KV buffer 的指针注册出去，让 prefill 直接写到 decode 的 buffer

**Decode 侧把 KV 挪到 host**

- `DecodeKVCacheOffloadManager.offload_kv_cache` 会把 decode 新生成的 KV 按 page 从 GPU 写到 host pool
- 这条路用的是 `HiCacheController.write(...)`

**社区对 NSA 的 offload 范围**

- 默认会同时 offload
  - `kv_cache`，每 token 每 layer 656B
  - `indexer_buffer`，每 token 每 layer 132B（`index_k_with_scale_buffer`）
- 对应实现是 `NSATokenToKVPoolHost`
  - `get_size_per_token` 会把 `indexer_buffer` 计入 host 内存开销
  - `load_to_device_per_layer` 与 `backup_from_device_all_layer` 除了搬 KV，还会搬 index
  - index 搬运要求 indices 按 `page_size` 对齐，否则会直接报错

### LRU 与淘汰（Prefix Cache）

- 社区的 LRU 主要服务于 `tree_cache` 的 prefix 复用，用来决定哪些 prefix 保留在 host 或从 host 回灌回 GPU
- 状态与决策在 CPU：`RadixCache` 的 `TreeNode.last_access_time` 等信息由 `evict_policy` 驱动，常用策略是 `LRUStrategy`
- 淘汰粒度跟 `page_size` 强绑定：prefix 的 `host_value` 是一段 host indices，`load_back` 也是按整段回灌；`HiCacheController.start_loading` 会对同一批 indices 加载所有 layers
- 这套 LRU 不解决 NSA decode 每层 `topk` 的 L2 命中问题，不能直接拿来替代的 GPU LRU

## 4. 迁移思路

### 融合后的流程

**Prefill**

保持社区 PD disaggregation 不变，prompt 的 `kv_cache` 与 `index_k_with_scale_buffer` 直接写在 decode 侧 GPU。

**Decode（仅 `--enable-kv-offload` 且 `attention_backend=nsa`）**

**基线约束（避免破坏社区默认路径）**  
不改模型真实 `page_size`（NSA 仍为 64），不把全局 `page_size` 改成 1。  
`topk_indices/page_table_1` 继续按 token 语义（`page_size=1` 逻辑表）驱动 sparse attention。

**step 0，layer 0**  
先用 GPU 全量 prompt KV 计算 attention。  
并行触发一次 D2H 备份，把 page 对齐的 prompt `kv_cache` 写入 host pool。  
`prompt_aligned_len = prompt_len // page_size * page_size`  
`host_indices = HiCacheController.write(device_indices=prompt_kv_indices[:prompt_aligned_len])`

**进入 layer 1 前**  
等待 D2H 完成。  
释放 GPU 上已备份的全量 prompt KV slot，后续只保留 `sparse_kv_buffer` 作为 L2。

**layer 1 及之后，每层循环**  
`Indexer` 基于 GPU `index_k_with_scale_buffer` 计算 `topk_indices`。  
对 `topk_indices` 做 `dedup` 并过滤无效值（`-1`），再用 `cpu_slot_to_gpu_slot[layer]` 判定 hit/miss。  
GPU LRU 计算 miss 并分配目标 GPU sparse slot，语义等价 `PrepareIntraLayerH2D` 输出的 `evict_ids`。  
H2D 只搬 miss token 的 `kv_cache`，写入 `sparse_kv_buffer`。  
复用 HiCache 现有按 indices 搬运能力（`kernel` 路径），不强依赖特定 H2D 算子。  
attention 在读取该层 KV 前等待 load 完成，不做 dual-attn 分流。

**每 step 结束**  
执行 `CompactCacheIdsBlock` + `ReorderOutCacheLocSparse`，压缩 LRU 状态并整理可用 slot。

**decode 增量 KV**  
新生成 token 的 `kv_cache` 以 page 对齐批次写入 host pool，写完即可提前释放 GPU 全量 KV slot。  
尾部不足 `page_size` 的部分保留在 GPU，直到凑满一页再备份（hicache 逻辑设定）。

### 搬运与同步细节

| 关注点 | 内容 |
| --- | --- |
| **page 对齐约束** | page 对齐只约束 host pool 的 `alloc`：`HiCacheController.write` 的 `device_indices` 长度必须是 `page_size` 的整数倍 |
| **H2D 粒度** | H2D 回填按 token 粒度做：KV-only host pool 的 `load_to_device_per_layer` 支持任意 indices，实际只拷 miss token，不需要把 miss 展开成整页 |
| **接口边界** | 不能直接复用 `HiCacheController.start_loading` 做 per-layer miss：它会对同一批 indices 加载所有 layers，和 NSA 每层不同的 `topk_indices` 不匹配 |
| **同步** | 同步建议沿用社区 `LayerDoneCounter`：每层 load 完成后 `complete(layer)`，attention 侧用 `wait_until(layer)` 保证读之前已完成 |
| **算子复用** | token miss 路径不新增底层算子，直接复用 HiCache 现有单层搬运接口：`mem_pool_host.load_to_device_per_layer(device_pool, host_indices, device_indices, layer_id, io_backend)`；`layer_first + kernel` 下该接口会走 `jit_transfer_hicache_one_layer` 或 `transfer_kv_per_layer` |
| **控制器改造** | 控制器侧只需新增 per-layer 调度接口（例如 `load_one_layer`），用于替代 `start_loading` 的全层循环，不改变默认路径行为 |

### 增量开发步骤

- [ ] **1. 增加开关与路由，不影响默认行为**：在 `server_args.py` 增加开关，例如 `--enable-kv-offload`；仅在 `attention_backend=nsa` 的 decode 路径启用；其他场景保持社区原逻辑。
- [ ] **2. 提供 NSA 的 KV-only host pool**：host pool 只承载 `kv_cache`，不承载 `index_k_with_scale_buffer`，避免 `NSATokenToKVPoolHost` 的 page 对齐限制；选用 `hicache_mem_layout=layer_first` + `hicache_io_backend=kernel`，保证 H2D 回填可以按 token indices 零散搬运；不把 token miss 能力建立在 `page_first_direct` 上，避免路径退化成 page 主导搬运。
- [ ] **3. 建立 per-request 映射与生命周期**：维护 token 到 host indices 的映射，以及 GPU L2 的 `cpu_slot_to_gpu_slot/cache_ids/...` 等 LRU 状态；请求结束时统一释放 host indices 与 GPU sparse slot，避免泄漏与交叉污染。
- [ ] **4. 在 NSA decode 内接入 per-layer token miss 回填**：per-layer：`Indexer -> dedup/过滤(-1) -> miss 计算与 LRU 更新 -> token级 H2D 回填 -> attention wait`；为不破坏社区 `HiCacheController.start_loading`（全层循环）语义，新增 per-layer 接口（例如 `load_one_layer`），仅在 `enable_kv_offload` 分支使用；decode miss buffer 使用 token 语义 allocator（`page_size=1`），与全局 KV pool 的真实 `page_size` 解耦。
- [ ] **5. 接入 LRU 后处理**：per-step：执行 `CompactCacheIdsBlock` 与 `ReorderOutCacheLocSparse`。
