# GB200 P/D 串通与 Mooncake KV 传输

状态: 已完成
截止日期: 04/18/2026
任务类型: FEATURE
小结: 控制面已基本串通（P返回首token、D enqueue、recv task提交），但 Mooncake 不支持 NVLINK
描述: 4/18: 手工两段式打通 → 4/18: Mooncake metadata修复 → 4/22: chat endpoint恢复+encoder fallback → 4/22: pdtest mock_ic P首token成功但send_tensors后segfault

## 任务描述

将 P 节点和 D 节点通过 Mooncake KV 传输串通，实现端到端 PD 分离推理。

## 子任务

- [x]  Mooncake metadata server 配置（etcd backend 对齐）
- [x]  Mooncake http metadata 插件与 etcd API 不兼容问题定位
- [x]  dp_affinity_scheduler round-robin 与 recv_requests broadcast 不匹配修复
- [x]  TpModelWorkerClient.get_memory_pool() 返回值分叉修复
- [x]  /v1/chat/completions 恢复（原硬编码 501）
- [x]  DeepSeek-V3.2 无 chat_template 的 fallback 补齐（adapter + encoder）
- [x]  offload IPC 后缀统一使用 node-local DP rank
- [x]  request/cache status EP 全组同步条件修正
- [x]  D-only debug cache-load no-op 修复
- [x]  P 真源脚本固定（DISABLE_TP_WITH_SP/ENABLE_TP_WITH_CP_NSA/ENABLE_SM_FP8_GEMM_1D1D）
- [ ]  Mooncake RDMA 在 GB200 上不可用（mlx5_bond_0 不存在 + registerRecvBuffer Bad address）
- [ ]  当前固定 MOONCAKE_USE_RDMA=FALSE 走 TCP
- [ ]  P 侧 Intern FlashMLA sparse prefill params.h_q != B_H 断言
- [ ]  P TP0 Mooncake send_tensors 后 UCX/libucs segmentation fault
- [ ]  端到端 PD 分离推理未完成

## 支持文件

Mooncake 源码确认：vllm_adaptor.cpp 只有 rdmq/tcp 两支，无 NVLink/IMEX/MNNVL backend