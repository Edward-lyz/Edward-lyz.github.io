# 1. 背景

模拟器是算子采集-消费的下游核心模块，之前由于代码量大，迁移后只验证了功能正确性，对代码结构，职责分配等没做仔细 review。目前需要推广出去让更多的人参与开发，那么就需要仔细 review + 优化，才方便进一步的推广。

---
# 2. 初步 review 结果

- [ ] 高: `prefill`(预填充) 路径大量使用 `args.prefill_seq_len`，但 args 构造与 CLI 只提供 `ctx_len`，会触发 `AttributeError` 或让吞吐计算用默认 4096，结果不可信。
- [ ] 高：绘图 `CLI` 定义了 `--moe-dtype`，但调用 `plot_simulator_results` 时完全没用这个参数，属于“传了也无效”
- [ ] 高: `mtp=false` 且 `config` 未提供 `mtp_step/accept_ratio` 时，结果文件名与错误日志直接索引 `config[...]` 会 `KeyError`。
- [ ] 中: 通过 `sys.path.insert` 进行相对路径导入，强依赖目录结构，打包兼容性差。
- [ ] 中: `simulator` 侧自建 `OpBackendManager/import_operator `等，与 `core` 的 `BackendManager/OperatorManager `并行，复用与一致性不足。
- [ ] 中: 核心计算链存在 `hard-coded(硬编码)` 时间常数（如 15、20），缺少配置化入口，扩展/校准成本高。