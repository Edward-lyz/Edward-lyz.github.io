# GB200-NVL72 aiak 代码适配

## 0 摘要

这篇文章记录把 aiak_sglang_offload 部署到 GB200 NVL72 集群上的全过程。从 4 月 8 日到 4 月 22 日，两周时间里，把一个原本只跑在 X86 + B200 上的 PD 分离推理框架，搬到 ARM64 + Blackwell GPU + 机架级 NVLink fabric 的全新硬件上。

当前结果：P 节点和 D 节点各自可以独立跑通推理，手工两段式 P→D 验证通过；全自动端到端 PD 分离还没走完，卡在 Mooncake `send_tensors` 触发 UCX segfault。

## 1 简介

目标是让 DeepSeek-V3.2 671B fp8 模型在 GB200 NVL72 上跑 PD 分离推理。P 节点做 prefill，D 节点做 decode，两者之间通过 Mooncake 传输 KV cache。

GB200 NVL72 是一个机架里 72 个 Blackwell GPU，分成 18 个计算托盘，每个托盘 4 个 GPU，跑各自 Linux OS。跨托盘 GPU 通过 MNNVL 直接访问对方显存，不需要走 PCIe 或 IB。

## 2 硬件认知

### 2.1 Fabric Manager 不在计算托盘上

GB200 NVL72 的 Fabric Manager 运行在 NVLink switch tray 的 NVOS 里，不是计算托盘上的 systemd 服务。计算托盘侧只需要确认 GPU driver 正常、nvidia-imex 运行、Fabric state = Completed。

### 2.2 MNNVL 不是传统 RDMA

GB200 上跨节点 GPU 通信走 NVLink fabric + nvidia-imex，不是 IB RDMA。DeepEP 通过 `allow_mnnvl=True, use_fabric=True` 走这条路径。Mooncake 源码只有 rdma 和 tcp 两个 backend，没有 NVLink/IMEX/MNNVL。

### 2.3 同批次机器环境不一致

同批交付的机器驱动版本、IB 网卡状态、nvidia-imex 运行状态都不一致。部署前必须检查驱动版本一致、imex 运行、Fabric state 正常、IB GID 兼容。

## 3 环境配齐

全组件需要源码编译并适配 CUDA 13 + ARM64 + sm_100：vllm 0.6.4、sgl-kernel、flashinfer、SpeedGate、DeepGEMM、fast_hadamard_transform、aiak 自定义算子。

关键环境点：

- `CMAKE_CUDA_ARCHITECTURES=100` 必须显式设。
- `NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5` 必须用逗号。
- `NCCL_SOCKET_IFNAME` 和 `NCCL_SOCKET_FAMILY=AF_INET` 要同时设置。
- 两台参与通信的机器都必须运行 nvidia-imex。

## 4 软件适配

主要修复：

- P 节点 forward：`tokenizer_manager.py` 的 `input_ids=None`、`apply_rope_with_cos_sin_cache_inplace` fallback、`fast_hadamard_transform` 缺失。
- D 节点 topology 纠偏：正确拓扑是 TP1/DP8，不是 TP8/DP1。
- offload 路径：stale JIT lock、`compact_cache_ids_block` int32、dp_affinity_scheduler、memory pool 返回值、`trans_stream` 漏传、offload 解包错位、`SET_DEEP_DP_MAX_TOKENS=256`、IPC 后缀、EP 组同步条件。
- `/v1/chat/completions` 恢复。
- DeepSeek-V3.2 无 chat_template 的 fallback。

## 5 当前状态

| 配置组合 | 状态 |
| --- | --- |
| P 节点 gb200-5+6 | 已跑通，curl 返回正常 |
| P 节点 gb200-1+0 | 已跑通，需 NCCL_IB_HCA + imex |
| D-only no-graph + offload + no dual | 最小请求返回 200 |
| D-only graph + offload + no dual | 最小请求返回 200，需 SET_DEEP_DP_MAX_TOKENS=256 |
| D-only graph + offload + dual | 首请求 crash，stage1 FlashMLA illegal memory access |
| 手工两段式 P→D | 已打通 |
| pdtest mock_ic | P 返回首 token，D enqueue 成功，P send_tensors 后 UCX segfault |
| 端到端自动 PD | 未完成 |

## 6 经验总结

先查环境再调代码。GB200 不是大号 HGX；Fabric Manager、MNNVL、NCCL 选网卡、nvidia-imex 都会改变排查方向。同批机器环境不一致是最大风险。JIT lock 文件也要作为稳定性问题处理，最好预编译或随镜像分发 so。
