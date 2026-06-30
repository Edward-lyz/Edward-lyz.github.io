# 【工程】kv_offload 迁移至社区 HiCache 初稿 (不考虑阿里前提下)

## 1. 结论

<aside>

当前 `kv_offload`是一种**主动的**、**per-layer** 、**Decode 侧重点优化**的KV 管理策略（每层 Indexer → miss 判定 → H2D 回填 → Attention），与社区 HiCache 的**被动的**，**按需加载、Prefill 重点优化的**路径（prefix 匹配 → 全层预取 `start_loading`）在设计目标和粒度上完全不同。因此，初步将 `kv_offload` 特性作为 Decode 侧的一个优化开关引入，且保持社区的代码风格

</aside>

### Decode 侧实现细节对照表

| 实现细节 | 当前 | 社区 | WIP |
| --- | --- | --- | --- |
| **Host pool 分配** | `NSATokenToKVPoolOffload`：同时分配好 CPU+GPU，KV 在 CPU pinned，index 在 GPU | `NSATokenToKVPoolHost`仅 HiRadixCache prefix 场景使用，**搬 kv+indexer**，**page 对齐，位于 CPU** | `NSAKVOnlyTokenToKVPoolHost` ，和 `SparseCacheController` 一一对应，社区layer-first 布局，只保存 KV cache，位于 CPU pinned |
| **GPU Pool 分配** | `sparse_kv_buffer`：独立连续 GPU buffer，代替 dense 语义的 `kv_buffer` | 无，KV 全在 `token_to_kv_pool` ，按需申请 slots | 无独立 `sparse_kv_buffer`；在 `NSATokenToKVPool.kv_buffer` 内按 request 申请连续 sparse slots（`alloc_contiguous`），请求结束后释放 |
| **`req_to_token` 语义** | 存 **CPU slot, Attn 不读取** | 存 **GPU slot， Attn 读取** | 保持 GPU device slot；新增 `req_seq_to_host_slot` 维护 `seq position -> host slot` 映射 |
| **`topk_indices` 语义** | 值即 CPU host slot，直接查 `cpu_slot_to_gpu_slot` | 值为序列位置（`0..seq_len-1`），需翻译为 host slot | 先 `map_seq_positions_to_host_slots`（seq pos -> host slot），再 `build_gpu_page_table`（host slot -> GPU slot） |
| **controller** | 自定义 CUDA kernel 直接操作，无 controller 语义 | `HiCacheController`：page 粒度，全层循环 | `SparseCacheController`：token 粒度，单层 H2D + per-layer D2H；LRU 更新逻辑在 `LRUCacheState`  |
| **LRU 状态** | GPU 侧 per-layer LRU（`cpu_slot_to_gpu_slot`、`cache_ids` 等） | `RadixCache` 的 CPU 侧 LRU（prefix 复用） | GPU 侧 per-layer 操作 |
| **D2H 触发** | Decode 阶段每 step per-layer 频率，以 token 为粒度写回 L2 | Decode 阶段，整个请求结束后为触发点，按 page 粒度写回 L2/L3 |   1. 首次 decode 先备份 全量 cache 到 host
  2. Decode 阶段每 step 增量 D2H，按 token 粒度、per-layer 执行 |
| **H2D 触发** | Decode 阶段每层 Indexer topk → miss → per-token H2D | Prefill 阶段 prefix 匹配后全量 `load_back` | Decode 阶段每层执行 `Indexer topk -> miss -> prepare_layer_swap -> load_tokens_h2d_one_layer` ，接入了当前的 H2D CUDA OP |
| **NSA backend 集成** | `OffloadArgs`、event 同步、per-layer 回填完整闭环 | 无 offload 逻辑 | 接入 `lru_state.prepare_layer_swap` + `build_gpu_page_table`，并通过 `decode_cache_controller` + CUDA event 保证层间同步 |
| **decode 增量 KV** | 每 step per-layer `set_kv_cpu_op` 写回 CPU | 请求结束时 page 对齐 offload | 每 step 在 `scheduler_output_processor` 按层调用 `transfer_kv_per_layer_mla` 方法，不新增OP |
| **后处理** | `CompactCacheIdsBlock` + `ReorderOutCacheLocSparse` | 无 | 接入 `lru_post_process`，内部执行 `compact + reorder`  |
| **server_args** | `enable_kv_offload` | `disaggregation_decode_enable_offload_kvcache` | `enable_decode_nsa_kv_offload` |

### 取舍

- **新增 `SparseCacheController`**：`HiCacheController` 粒度是 page 级别（`start_loading` 对同一批 indices 循环所有 layers），效率低且语义不匹配 per-layer token miss 回填。且阿里最新的 Roadmap 也提到要新增该 Controller，说明是有必要的。

[[Roadmap-HiCache]: Unified Storage Framework for SGLang Across Diverse Scenarios · Issue #18239 · sgl-project/sglang](https://github.com/sgl-project/sglang/issues/18239)

- **新增 KV-only host pool**：社区 `NSATokenToKVPoolHost`（仅用于 HiRadixCache prefix 场景）同时搬 kv_cache 和 indexer_buffer，且 indexer 搬运要求 indices 按 page_size 对齐——与 per-token sparse 回填冲突；`DecodeKVCacheOffloadManager` 不感知 NSA，无对应 host pool
- **新增 `LRUCacheState`**：社区 `RadixCache` 的 LRU 是 CPU 侧 prefix 复用，无法解决 NSA decode 每层 topk 的 L2 命中问题，引入独立文件，做增量开发
- **GPU 上需要独立的 sparse buffer**：对齐 `sparse_kv_buffer`，H2D 直接写入连续 buffer，attention 从中读取
- **不引入 L3 storage，不做 dual attention 优化代入**

---

## 2. 当前的实现逻辑

### 数据结构

**NSATokenToKVPoolOffload** (`memory_pool.py`)

- 继承自 `NSATokenToKVPool`
- `kv_buffer`: `List[Tensor]`，per-layer，shape `[size+page_size, 1, 656]`，**CPU pinned memory**
- `sparse_kv_buffer`: `List[Tensor]`，per-layer，shape `[int(size*cache_ratio)//page_size*page_size, 1, 656]`，**GPU 连续**
- `index_k_with_scale_buffer`: per-layer，**始终在 GPU**，Indexer 不需等待 host load

**ReqToTokenPool 的 LRU 状态**（`enable_kv_offload=True` 时额外分配）

- `cpu_slot_to_gpu_slot`: `[num_layers, max_total_tokens]`，host slot → GPU sparse slot，`1` 表示不在 GPU
- `cache_ids`: `[num_layers, pool_size, max_cache_tokens_per_req]`，每个请求的 LRU 列表
- `cache_ids_inverse`: `[num_layers, max_total_tokens]`，反查表
- `cache_lens`: `[num_layers, pool_size]`，每个请求的 LRU cache 长度
- `out_cache_loc_sparse`: `[num_layers, pool_size, max_cache_tokens_per_req]`，可用 GPU sparse slot
- `cache_missed_counts`: `[num_layers, pool_size]`

### Offload 数据范围（attention_backend=nsa）

- 仅 offload `kv_cache`：每 token 每 layer 656B
- **不 offload** `index_k_with_scale_buffer`：每 token 每 layer 132B
- 结果：Indexer 基于 GPU 常驻 index 做 topk，无需等待 host load

### Prefill 侧

1. prefill 结束后调 `send_kv_cache` → `MooncakeKVTransferAgent.send_kv_caches(...)`
2. 开 `enable_kv_offload` 时额外传 `select_sparse_kv_indices`（prompt 前 `buffer_size` 个位置）
3. 发送后释放 prefill 侧 KV

### Decode 侧接收

1. `recv_kv_caches(loc=out_cache_loc)` 把 prompt KV 收到 decode 侧 CPU `kv_buffer`
2. `UpdateReqToTokenPool` 初始化 LRU 状态：`cpu_slot_to_gpu_slot`、`cache_ids`/`cache_ids_inverse`/`cache_lens`、`out_cache_loc_sparse`

### Decode 每步每层（deepseek_v2.py）

1. **Indexer** → `topk_indices`
2. **等待前一层写 CPU 完成**：`wait_event(set_kv_cpu_finish_event_list[layer_id-1])`
3. **miss 计算 + LRU 更新**：
    - LRU 模式：`PrepareIntraLayerH2D` → `CalcIntraLayerTransTopkIds` → `UpdateIntraLRUCacheAndGetEvictIds`
    - 非 LRU：`PrepareIntraLayerH2DNoLRU`（全部当 miss）
    - 输出 `evict_ids: [bs, topk, 2]`
4. **H2D 回填**（`trans_stream`）：`IntraLayerH2D` — 自定义 CUDA kernel，CPU pinned → GPU sparse_kv_buffer
5. **Attention 等待**：`OffloadArgs.intra_layer_h2d_finish_event`
6. **写回 CPU**：`SetKVCPU` per-layer 写入 CPU `kv_buffer`

### 每步后处理

- `CompactCacheIdsBlock`：压缩 `cache_ids` 空洞
- `ReorderOutCacheLocSparse`：重排可用 GPU slot

---

## 3. 社区实现逻辑

### HiCacheController (`cache_controller.py`)

- **唯一的** controller，管理 GPU ↔ host KV 搬运
- **page 级别粒度**：
    - `write(device_indices)` → host `alloc(len)` 要求 `need_size % page_size == 0`
    - `start_loading()` → 对同一批 indices **循环所有 layers**（`for i in range(self.layer_num)`）
- `LayerDoneCounter`：per-layer 完成通知，用于 prefix 回灌的 layer overlap
- 独立 `write_stream` / `load_stream`
- 可选挂载 storage backend（L3）

### NSATokenToKVPoolHost (`memory_pool_host.py`)

- 继承 `MLATokenToKVPoolHost`
- **仅在 `HiRadixCache`（prefix cache）场景中使用**，`DecodeKVCacheOffloadManager` 不感知 NSA
- **同时承载** `kv_cache` 和 `index_k_with_scale_buffer`
- **indexer 搬运要求 indices 按 `page_size` 对齐**（`_get_indexer_page_indices` 中 `numel() % page_size != 0` 直接 raise）
- 这与 per-token sparse 回填存在**冲突**

### DecodeKVCacheOffloadManager (`decode_kvcache_offload_manager.py`)

- 仅在 `disaggregation_decode_enable_offload_kvcache` 时启用
- 使用 `HiCacheController`，**不感知 NSA**
- `isinstance` 只检查 `MHATokenToKVPool` 和 `MLATokenToKVPool`；`NSATokenToKVPool` 继承 `MLATokenToKVPool`，会 fallback 到 `MLATokenToKVPoolHost`（不搬 indexer，但也不搬 sparse buffer）
- 功能仅为请求结束时把 decode 增量 KV 写到 host，然后触发 L3 backup
- **没有** prompt D2H、per-step incremental offload、LRU 状态管理

### Prefix 匹配和回灌

- `tree_cache.match_prefix(...)` → `host_hit_length` → `load_back` 把整段 prefix 搬回 GPU
- 粒度是整段 prefix、全层循环，为 **prefix 复用**设计

### NSA 后端 decode 路径（nsa_backend.py）

- `SGLANG_NSA_FUSE_TOPK=True` 时 `page_table_1 = topk_indices`
- 否则 `transform_index_page_table_decode` 做页表转换
- **没有任何 offload 相关逻辑**

---

## 4. 迁移思路

### 4.1 代码架构

**新增组件**

```
新增（在社区框架上增量开发）
├── SparseCacheController              # 新文件或追加到 cache_controller.py
│   ├── offload_tokens_d2h()           # 全层 D2H（write_stream）
│   └── load_tokens_h2d_one_layer()    # 单层 H2D（load_stream）
├── NSAKVOnlyTokenToKVPoolHost         # 追加到 memory_pool_host.py
│   └── 继承 MLATokenToKVPoolHost，省 ~16.75% host 内存
├── LRUCacheState                      # 新文件 lru_cache_state.py
│   ├── prepare_layer_swap()           # miss→LRU→H2D（per-layer）
│   ├── build_gpu_page_table()         # topk→GPU slot 翻译
│   └── lru_post_process()            # compact+reorder（per-step）
└── server_args.py
    ├── enable_decode_nsa_kv_offload: bool
    ├── kv_offload_cache_ratio: float
    └── max_cache_tokens_per_req: int
```

**修改组件**

```
修改（最小修改）
├── DecodeKVCacheOffloadManager        # 扩展 NSA 分支
│   ├── 新增 backup_prompt_kv()        # 首次 decode 前 prompt D2H
│   ├── 新增 sync_prompt_backup()      # 同步等待 D2H
│   ├── 新增 offload_decode_kv_incremental()
│   └── 新增 lru_post_process()
├── nsa_backend.py _forward_decode()   # 接入 per-layer H2D + GPU page table
├── scheduler.py                       # 挂载 LRU、调 prompt D2H、per-step 后处理
└── memory_pool.py ReqToTokenPool      # 挂载 lru_state / decode_cache_controller
```

**调用关系**

```
Scheduler
 └── DecodeKVCacheOffloadManager（扩展）
      ├── SparseCacheController（新增）
      │    ├── offload_tokens_d2h()   → 全层 D2H（write_stream）
      │    └── load_tokens_h2d_one_layer() → 单层 H2D（load_stream）
      ├── NSAKVOnlyTokenToKVPoolHost（新增）→ host pool (layer_first)
      └── LRUCacheState（新增）
           ├── prepare_layer_swap()   → miss→LRU→H2D (per-layer)
           ├── build_gpu_page_table() → topk→GPU slot
           └── lru_post_process()     → compact+reorder (per-step)

nsa_backend.py decode path
 └── per-layer 循环:
      1. Indexer → topk_indices（GPU 常驻 index_k_with_scale_buffer）
      2. lru_state.prepare_layer_swap(layer_id, topk_indices)
      3. page_table_1 = lru_state.build_gpu_page_table(...)
      4. Attention(kv_cache, page_table_1)
```

### 4.2 整体流程

**Prefill 侧**：保持社区 PD disaggregation 不变。prompt 的 `kv_cache` 与 `index_k_with_scale_buffer` 直接写在 decode 侧 GPU。

**请求首次进入 decode：prompt D2H**

1. `backup_prompt_kv(req)` → `SparseCacheController.offload_tokens_d2h()` → 把 prompt `kv_cache` 写入 host pool（只搬 kv，不搬 indexer）
2. `sync_prompt_backup(req)` → 同步等待 D2H 完成
3. 释放 GPU 上已备份的 prompt KV slot
4. `LRUCacheState.update_req_seq_host_slots()` → 记录序列位置 → host slot 映射

**每 step decode forward（per-layer 循环）**

1. **Indexer** → `topk_indices`
2. **`prepare_layer_swap(layer_id, topk_indices, forward_batch)`**：
    - 序列位置 → host slot 映射
    - CUDA kernel 计算 miss + 更新 LRU + 输出 `evict_ids`
    - `load_tokens_h2d_one_layer()` 异步搬运 miss token KV
    - 同步等待 H2D 完成
3. **`build_gpu_page_table()`** → topk 翻译成 GPU slot
4. **Attention** 读取 `sparse_kv_buffer` + `page_table_1`

**每 step 结束**

1. `offload_decode_kv_incremental(req)` → 新生成 token KV D2H 到 host
2. `lru_post_process()` → `CompactCacheIds` + `ReorderOutCacheLocSparse`

**请求结束**

1. offload 剩余 KV → 释放 GPU slot → 释放 host indices → `clear_req()` 清理 LRU 状态

---

## 5. TODO List

- [x]  **新增 `server_args` 开关**：`enable_decode_nsa_kv_offload`、`kv_offload_cache_ratio`、`max_cache_tokens_per_req`；仅在 `attention_backend=nsa` 的 decode 路径生效
- [x]  **新增 `NSAKVOnlyTokenToKVPoolHost`**：继承 `MLATokenToKVPoolHost`，KV-only（不搬 indexer），`layer_first` 布局
- [x]  **新增 `SparseCacheController`**：独立于 `HiCacheController`，提供 token 粒度 `offload_tokens_d2h` 和 per-layer `load_tokens_h2d_one_layer`
- [x]  **新增 `LRUCacheState`**：per-layer GPU 侧 LRU 状态 + CUDA kernel（miss 计算、LRU 更新、compact、reorder）
- [x]  **扩展 `DecodeKVCacheOffloadManager`**：NSA 分支使用上述组件，新增 `backup_prompt_kv` / `sync_prompt_backup` / `offload_decode_kv_incremental` / `lru_post_process`
- [x]  **接入 `nsa_backend.py` decode 路径**：`prepare_layer_swap` + `build_gpu_page_table` 嵌入 per-layer 循环
- [x]  **接入 `scheduler.py`**：挂载 LRU 状态到 `req_to_token_pool`，请求首次 decode 时 prompt D2H，每 step 结束调后处理
- [ ]  **性能优化**：异步 H2D event 等待、MTP 支持