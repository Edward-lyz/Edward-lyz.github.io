# 【工程】模拟器时间查询 python 优化

## 1. 问题背景

simulator_v2 是 aiak_ds_tool 库中的推理性能模拟器，给定模型配置和运行参数，通过查询预采集的算子性能数据库，输出逐层/逐算子的耗时预估。

核心调用链：

```
SimulatorV2App.export_result()
  └─ SimulatorEngine.simulate()          # 对每个 case 逐层模拟
       └─ OperatorRegistry.get_time()    # 对每个算子查询耗时
            └─ DBEngine.query()          # 打开 sqlite3 DB → 执行查询 → 返回行
```

问题：16-case prefill 场景下 wall time = 32.9s。对一个纯 CPU 数据查询+计算工具而言，显然不可接受。

## 2. Trace 工具选择：cProfile vs pyinstrument

| 维度 | cProfile | pyinstrument |
| --- | --- | --- |
| 采样方式 | 确定性（每次调用 hook） | 统计采样（~1ms 间隔） |
| 开销 | 5-30%（侵入式） | ≈2%（低干扰） |
| 输出 | flat 表：ncalls / tottime / cumtime | 调用树 + 按时间折叠 |
| 强项 | 精确调用次数，量化单次开销 | 快速定位热路径层级关系 |
| 弱项 | 输出扁平，大量 import 噪声 | 丢失低频调用细节 |
| 可视化 | snakeviz / speedscope | 自带 HTML / speedscope JSON |

实战中两者互补：先用 pyinstrument 看树状结构找到热路径，再用 cProfile 量化该路径上的调用次数和单次开销。

### 命令示例

```bash
# cProfile: 输出 pstats 二进制文件
python3 -m cProfile -o /tmp/simv2.pstats -m aiak_infer_tools simulator_v2 \
  --model-config ... --run-config ... --export-output /tmp/out.csv

# pyinstrument: 输出 HTML 调用树
pyinstrument -r html -o /tmp/profile.html \
  -m aiak_infer_tools simulator_v2 ...

# pyinstrument: 输出 speedscope JSON
pyinstrument -r speedscope -o /tmp/profile.json \
  -m aiak_infer_tools simulator_v2 ...
```

## 3.  基线测量

测量环境：macOS / Apple Silicon / Python 3.11 / sqlite3 本地文件

| 场景 | Wall Time | cProfile 下 | DB query 调用次数 |
| --- | --- | --- | --- |
| decode 1 case | — | 3.52s | 2,516 |
| prefill 16 cases | 32.9s | 35.8s | 45,424 |

<aside>

关键发现：prefill 16 cases 生成 42,960 个算子请求，但 unique DB query key 仅 556 个。

</aside>

77 倍重复查询。大部分时间花在重复打开相同的 sqlite3 文件、执行相同的 SQL、转换相同的行。这是典型的 memoization 机会。

## 4.  瓶颈诊断

从 pyinstrument baseline 调用树提取的热路径拆解：

| 调用路径 | 耗时 | 占 simulate 比例 |
| --- | --- | --- |
| `Connection.execute`（获取表名） | 9.0s | 33% |
|  `sqlite3.connect`（打开连接） | 1.7s | 6% |
| `Connection.close` | 1.3s | 5% |
| `Cursor.execute`（实际查询） | 2.2s | 8% |
| `_resolve_backend_name`（listdir） | 1.7s | 6% |
|  `_convert_row` + `listcomp` | 2.0s | 7% |
| `DBEngine.query` 总计 | ~22s | 81% |
| `dataclasses.asdict`（payload 序列化） | 4.7s | 17% |
| `asdict` on query list | 2.0s | 7% |

诊断结论：

1. **DB 层**：每次 `query()` 都重新 connect → 查 table name → 执行 SQL → close。45,424 次调用，每次 ~0.5ms。
2. **序列化层**：`dataclasses.asdict` 使用 `copy.deepcopy`，嵌套越深越慢。
3. **通信算子插值**：`interpolation_coordinate_rows` 对同一组坐标反复查 DB 获取邻近点。

## 5.  优化1：DB 查询缓存

很自然地，我们会想到对查询的结果进行缓存：

在 `DBEngine` 类上引入两层 class-level dict 缓存：

1. `_QUERY_DB_PATH_CACHE`：缓存 `(data_dir, device, op, backend, version) → db_path` 映射
2. `_QUERY_CACHE`：缓存 `(db_path, mtime_ns, size, filters)` → 查询结果行（以 tuple of tuples 存储，避免 mutable aliasing）

核心实现要点：

```python
# 缓存 key 包含 mtime_ns + size 作为 invalidation 信号
db_stat = os.stat(db_path)
query_cache_key = (
    db_path,
    db_stat.st_mtime_ns,
    db_stat.st_size,
    tuple(sorted((filters or {}).items())),
)
cached_rows = cls._QUERY_CACHE.get(query_cache_key)
if cached_rows is not None:
    return [dict(row_items) for row_items in cached_rows]

# cache miss → 正常查询 → 存入 cache
rows = [cls._convert_row(dict(row)) for row in cursor.fetchall()]
cls._QUERY_CACHE[query_cache_key] = tuple(tuple(row.items()) for row in rows)
```

cache value 用 `tuple(tuple(row.items()))` 而不是直接存 list of dict — 防止调用方修改 dict 后污染 cache。返回时用 `[dict(row_items) for row_items in cached_rows]` 重建新 dict。

Invalidation：在 `write()` / `delete()` / `batch_insert()` 之后调用 `cls.clear_query_cache()`。

**效果：**

| 指标 | Before | After Pass 1 | 提升 |
| --- | --- | --- | --- |
| prefill 16 cases wall | 32.9s | 5.5s | 6x |
| simulate-only wall | 29.6s | 4.9s | 6x |
| cache hit rate | — | 44,867/45,424 = 98.8% | — |

## 6.  优化 2：算子上层结果缓存

在 `OperatorRegistry.get_time()` 层面，对相同 (runtime context, op_time_request) 缓存最终的 `OpTimeResult`。

观察：即使 DB 缓存命中，每次 `get_time` 仍需执行 `build_time_request` → 构造 cache key → 查 DB → 后处理。对相同算子+相同 shape，结果必然相同。

```python
# OperatorRegistry.get_time() 内部
time_cache_key = self.time_cache_key(runtime, op_time_request)
op_time_result = self.operator_time_results.get(time_cache_key)
if op_time_result is None:
    queried_request, op_time_result = operator_class.get_time(...)
    self.operator_time_results[time_cache_key] = op_time_result
return op_time_request, op_time_result
```

cache key 设计：

```python
@staticmethod
def time_cache_key(runtime, op_time_request):
    backend, backend_version = resolve_backend(runtime, op_time_request)
    return (
        runtime_string_option(runtime, "data_dir"),
        runtime_string_option(runtime, "device_name"),
        backend,
        backend_version,
        op_time_request.concrete_op_name,
        tuple(sorted(op_time_request.get_time_kwargs.items())),
    )
```

**效果：**

| 指标 | After Pass 1 | After Pass 2 | 提升 |
| --- | --- | --- | --- |
| prefill 16 cases wall | 5.5s | 3.4s | 1.6x |
| OperatorRegistry.get_time | 3.8s | 1.5s | 2.5x |

## 7.  优化 3：查询键缓存

缓存 `build_time_request` 中的 `query_key_for_op` 结果。该函数通过 `OperatorManager.create_operator` 查找算子注册表，涉及字符串匹配和对象创建。

**效果：** 3.4s → 3.3s（微小收益，但消除了 profile 中 `build_time_request` 下的 `create_operator` 热点）。

## 8.  优化 4：运行时类型检查

`resolve_backend` 中使用 `isinstance(x, Mapping)` 做 dict-like 类型检查。Python 的 `typing.Mapping` 走 `__instancecheck__` → `__subclasscheck__` 链，每次调用开销远高于直接 `isinstance(x, dict)`。

改为 `from collections.abc import Mapping as MappingABC` 后，ABC 注册的 virtual subclass 检查路径更短。

```python
# Before: typing.Mapping（每次走 _GenericAlias.__instancecheck__）
from typing import Mapping
isinstance(x, Mapping)

# After: collections.abc.Mapping（直接 ABC 注册检查）
from collections.abc import Mapping as MappingABC
isinstance(x, MappingABC)
```

在热路径上避免 `typing` 模块的运行时类型检查。`typing.Mapping` 的 `isinstance` 比 `collections.abc.Mapping` 慢约 3-5x，因为前者需要经过 `_SpecialGenericAlias.__instancecheck__` → `__subclasscheck__` 的间接路径。

## 9. 方法论总结

### 性能分析四步法

**Step 1: 建立 baseline**
用 wall time 测基准值。如果 cProfile 开销大（>10%），优先用 pyinstrument。记录精确的调用次数。

**Step 2: 定位热路径**
pyinstrument 调用树找到占比最大的叶子节点。关注 cumtime 占总时间 >30% 的路径。

**Step 3: 量化重复度**
用 cProfile 的 ncalls 或手动插桩统计：unique 输入有多少？duplicate factor 多大？这决定了 cache 的收益上界。

**Step 4: 逐层优化 + 验证**
从最底层（IO/外部调用）开始缓存，逐层向上。每一层优化后：(a) 验证正确性，(b) 重新 profile 确认热点转移。

### Python 常见 CPU 陷阱

| 陷阱 | 症状 | 修复 |
| --- | --- | --- |
| 重复 sqlite3.connect() | connect/close 在 profile 中占 10%+ | 连接池或查询结果缓存 |
| dataclasses.asdict | _asdict_inner + copy.deepcopy 调用爆炸 | 避免在热路径序列化；用 tuple/namedtuple 代替 |
| typing.isinstance | _SpecialGenericAlias.__instancecheck__ 高频 | 用 collections.abc 对应类型 |
| dataclasses.replace | 每次创建新对象 + deepcopy fields | 如果 field 未变，直接重用原对象 |
| os.listdir 在循环中 | 系统调用反复扫描目录 | 结果缓存或预加载目录清单 |

### Cache 设计 checklist

1. **Key 完备性**：所有影响结果的输入都必须在 key 中。遗漏一个维度 = 返回错误结果。
2. **Value 隔离**：cache value 必须不可变或在返回时复制。避免调用方修改后污染。
3. **Invalidation**：所有写操作路径必须清缓存。宁可多清不可少清。
4. **容量控制**：无限增长的 cache 是内存泄漏。设 max entries 或用 LRU。
5. **作用域**：class-level（进程级）vs instance-level vs 请求级。选择匹配生命周期的作用域。

### 优化效果全景

| 阶段 | Wall Time | 相对 Baseline | 主要改动 |
| --- | --- | --- | --- |
| Baseline | 32.9s | 1.0x | — |
| Pass 1: DB cache | 5.5s | 6.0x | query rows + db_path 缓存 |
| Pass 2: OpTime cache | 3.4s | 9.7x | 算子最终结果缓存 |
| Pass 3: query key cache | 3.3s | 10.0x | operator lookup 缓存 |
| Pass 4: ABC isinstance | 3.3s | 10.0x | typing → collections.abc |

最终效果：10x 加速，输出 bit-exact。全部测试通过。无功能变更，纯性能优化。

## Quiz: 检验理解

- Q1: 为什么 DB 缓存用 mtime_ns + size 作为 invalidation key？
    
    > 因为 simulator 运行期间理论上不会有外部进程修改 DB 文件，但如果文件被修改（比如重新采集数据后再次运行），mtime_ns 和 size 的变化能确保不会读到旧缓存。这比简单的 path-based key 多一层安全保障，同时避免了 file hash 的计算开销。
    > 
- Q2: Pass 2 的 time_cache_key 为什么不直接用 op_time_request 对象作为 key？
    
    > dataclass 对象的 hash 默认基于 id()（如果未设 frozen=True），即使内容相同也不会命中。即使设了 frozen，嵌套 dict field 不可 hash。所以必须手动提取影响结果的字段构造 tuple key。
    > 
- Q3: 如果 simulator 改为 long-running server 模式，cache 设计需要怎么调整？
    
    > 需要引入 TTL 或 request-scoped cache。class-level 无限期缓存在 server 中会导致 stale data（DB 文件更新后仍返回旧结果）。可选方案：(a) 每次请求开始时 clear_query_cache；(b) 用 mtime check 做 lazy invalidation（当前实现已包含）；(c) 加 max_entries + LRU 防止内存无限增长。
    >