# 【工程】GB200-NVL72 aiak 代码适配

# 0 摘要

这篇文章记录了把 aiak_sglang_offload 部署到 GB200 NVL72 集群上的全过程。从 4 月 8 日到 4 月 22 日，两周时间里，我们把一个原本只跑在 X86 + B200上的 PD 分离推理框架，搬到了 ARM64 + Blackwell GPU + 机架级 NVLink fabric 的全新硬件上。最终结果是：P 节点和 D 节点各自可以独立跑通推理，手工两段式 P→D 也验证通过，但全自动端到端 PD 分离还没走完，当前卡在 Mooncake 的 send__tensors 触发 UCX segfault（mooncake 不支持 NVLink 协议）

过程中踩的坑分三类：硬件认知（GB200 不是大号的 HGX，MNNVL 的运行模型和传统 RDMA 完全不同）、环境配齐（驱动版本、IB 网卡状态、nvidia-imex，三件事必须同时到位否则怎么调都不对）、软件适配（从 vllm 到 sgl-kernel 到 DeepGEMM，几乎所有 CUDA 组件都要在 sm_100 + arm64 + CUDA13 上重新编译，aiak 代码本身也有大量接口分叉和 offload 路径的 bug）。

这篇文章把这些踩坑经验按时间线展开，尽量把每个问题的根因和排查思路写清楚，而不是只列现象。

---

# 1 简介

## 1.1 我们要做什么

目标是让 DeepSeek-V3.2 671B fp8 模型在 GB200 NVL72 上跑 PD 分离推理。P 节点做 prefill，D 节点做 decode，两者之间通过 Mooncake 传输 KV cache。这东西在 B200 上已经跑过了，代码是现成的（aiak_sglang_offload 分支），但在 GB200 上从来没验证过。

## 1.2 GB200 NVL72 是什么

一句话：一个机架里有 72 个 Blackwell GPU，分成 18 个计算托盘，每个托盘 4 个 GPU，跑各自的 Linux OS。GPU 之间通过 NVLink switch 互连，形成一个 NVLink domain。跨托盘的 GPU 可以通过 MNNVL（Multi-Node NVLink）直接访问对方的显存，不需要走 PCIe 或 IB。

这和传统的 HGX/DGX 有本质区别：HGX 是一个 OS 管理所有 GPU，NVLink 在单机内部；GB200 NVL72 是多个 OS 共享一个机架级 NVLink fabric。

## 1.3 工作量有多大

粗略统计：

- 编译/构建：7 个组件从源码编译，全部要适配 CUDA 13 + ARM64 + sm_100
- 代码修复：超过 15 处 bug fix，涉及 tokenizer_manager、parallel_state、memory_pool、nsa_backend、compact_cache、scheduler、data_parallel_controller、tp_worker_overlap_thread、encoder、openai_adapter 等模块
- 环境排查：14 台机器逐一检查驱动版本、IB 网卡状态、nvidia-imex、Fabric state
- 配置发现：NCCL_IB_HCA 必须用逗号分隔（分号不行）、NCCL_SOCKET_IFNAME + AF_INET 必须同时设、SET_DEEP_DP_MAX_TOKENS 默认 32 太小要改 256、USE_CUTLASS_FP8_BLOCK_GEMM 必须关掉

没有 K8S，没有自动化部署，全靠 SSH + docker exec + 肉眼看日志。

---

# 2 硬件认知

## 2.1 Fabric Manager 不在计算托盘上

我一开始按 HGX 的经验去找 nvidia-fabricmanager 服务，没找到，以为 MNNVL 不工作。后来查了 NVIDIA 官方文档才知道：GB200 NVL72 的 Fabric Manager 运行在 NVLink switch tray 的 NVOS 里，不是计算托盘上的 systemd 服务。你不需要（也没法）在计算托盘上安装它。

计算托盘侧只需要确认三件事：GPU driver 正常、nvidia-imex 运行、Fabric state = Completed。这就够了。

## 2.2 MNNVL 不是传统 RDMA

GB200 上跨节点 GPU 通信走的是 NVLink fabric + nvidia-imex，不是 IB RDMA。DeepEP 通过 `allow_mnnvl=True, use_fabric=True` 走这条路径。我们实测 2MPI×4GPU 可以跑到约 206 GB/s，确认 MNNVL 确实在工作。

但 Mooncake 不走这条路。它的源码里只有 rdma 和 tcp 两个 transport backend，没有 NVLink/IMEX/MNNVL。所以 Mooncake RDMA 失败不等于 GB200 的高速链路不可用，只是 Mooncake 还没适配这条路径。

## 2.3 同批次机器环境不一致

这是踩坑最大的教训。14 台同批交付的 GB200 机器，驱动版本、IB 网卡状态、nvidia-imex 运行状态都不一样：

- gb200-2 和 gb200-15 的 GPU 驱动是 580.x，其余是 570.172.08。580.x 的 imex 版本 570.x 不匹配，imex 启动不了
- gb200-1 的 mlx5_2 和 mlx5_6 是 Active（NVLink fabric 接口），只有 link-local GID；其他机器这两个 NIC 是 DOWN
- gb200-0 的 nvidia-imex 没有自动启动

gb200-5/6 恰好三个条件都满足，所以第一次就能跑。换到别的机器对就各种炸，不是 MNNVL 本身的问题，是环境没对齐。

部署前必须跑前置检查清单：驱动版本一致、imex 运行、Fabric state 正常、IB GID 兼容。不跑这个清单就直接试，只会浪费时间。

---

# 3 环境配齐

## 3.1 全组件源码编译

aiak_sglang_offload 依赖 vllm 0.6.3~0.6.4，但在 CUDA 13 + ARM64 上没有预编译 wheel。只能源码编译，而且不只是 vllm，几乎所有 CUDA 组件都要重编：

| 组件 | 编译方式 | 关键参数 |
| --- | --- | --- |
| vllm 0.6.4 | 源码 pip install | TORCH_CUDA_ARCH_LIST=10.0，补 Blackwell 架构支持 |
| sgl-kernel | 源码 cmake + pip | CMAKE_CUDA_ARCHITECTURES=100，MAX_JOBS=144 |
| flashinfer | JIT | FLASHINFER_CUDA_ARCH_LIST=10.0 |
| SpeedGate | 源码 | 补 10.0 compute capability |
| DeepGEMM | JIT | 默认即可 |
| fast_hadamard_transform | 源码 pip install | TORCH_CUDA_ARCH_LIST=10.0 |
| aiak 自定义算子 | JIT / 源码 | sgl_per_tensor_quant_fp8 fallback |

基础镜像用的 [nvcr.io/nvidia/pytorch:26.01-py3。编译过程本身没太多坑，主要就是](http://nvcr.io/nvidia/pytorch:26.01-py3。编译过程本身没太多坑，主要就是) CMAKE_CUDA_ARCHITECTURES=100 必须显式设，不然 cutlass 会尝试编译 sm_70 等不支持的架构然后炸掉。

4 月 22 日又做了一次运行时依赖固定：把 vllm/flashinfer/speedgate/sgl_kernel/deep_gemm/mscclpp 全部安装到 /usr/local/lib/python3.12/dist-packages，不再依赖 /opt/aiak 作为 PYTHONPATH。镜像约 15GB，已分发到 gb200-0 和 gb200-1。

## 3.2 IB GID 不匹配

gb200-1 的 mlx5_2 和 mlx5_6 是 NVLink fabric 接口，PORT_ACTIVE 但只有 link-local GID（fe80::...）。NCCL 自动选网卡时会选中这些 NIC，然后和远端的 IPv4-mapped GID（::ffff:33.0.x.x）对不上，ibv_modify_qp 报 EINVAL。

修复：`export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5`。注意分号和逗号的问题——当前运行时里必须用逗号分隔，分号会导致 NCCL 退回 Socket 然后尝试连 link-local IPv6。

同时还要设 `NCCL_SOCKET_IFNAME=enP5p9s0` 和 `NCCL_SOCKET_FAMILY=AF_INET`，否则 socket 层也可能走错。

## 3.3 nvidia-imex

deep_ep 的 `Buffer(allow_mnnvl=True, use_fabric=True)` 依赖 nvidia-imex 做跨节点 GPU memory import/export。imex 不运行就报 `operation not supported`。

两台参与通信的机器都必须运行 imex。gb200-0 的 imex 没有自动启动，需要手动 `systemctl start nvidia-imex`。

gb200-2 的 imex 启动不了，因为驱动 580.126.20 和 imex 包 570.211.01 版本不匹配。这种机器暂时不能用。

## 3.4 驱动版本

集群里混了 570.172.08 和 580.x 两种驱动。580.x 的机器（gb200-2、gb200-15）因为 imex 版本不匹配，MNNVL 路径不可用，目前只能用 570.172.08 的机器。

部署时两台机器的驱动版本必须一致，否则 NCCL 和 deep_ep 都可能出问题。

---

# 4 软件适配

## 4.1 P 节点 forward 链路修复

P 节点第一次跑起来就遇到三个问题：

1. `tokenizer_manager.py`：input_ids=None 导致 TypeError。补了 None 检查。
2. `sgl_kernel/__init__.py`：apply_rope_with_cos_sin_cache_inplace 的 fallback 不可用。在 GB200 覆盖版里补了。
3. `fast_hadamard_transform` 缺失。NSA indexer 的 rotate_activation 依赖它，源码编译安装。

修完这三个之后，P 节点 curl max_tokens=1 能正常返回。单次 forward 约 30 秒，包含首次 JIT 编译开销。

另外 `USE_CUTLASS_FP8_BLOCK_GEMM=0` 是 P 节点 completion 全流程可用的必要开关，否则 block-fp8 linear 会命中 deep_gemm.fp8_gemm_nt 的 `Unsupported architecture or scaling factor types`。

## 4.2 D 节点 topology 纠偏

D 节点的正确拓扑是 TP1/DP8，不是 TP8/DP1。旧文档里写的 TP8/DP1 是错的——按那个启动会把 world_size 算成 64，直接报 `CUDA error: invalid device ordinal`。

后来又试了 `--tp-size 8 --dp-size 8 --enable-dp-attention`，结果 model_runner 按 tp_size × dp_size 计算 world_size=64，同样是 invalid device ordinal。

确认是 TP1/DP8 之后，D 节点才能正常启动。

## 4.3 offload 路径 bug 修复

offload 路径的 bug 比较多，按时间顺序列主要的：

**reset_kv_[state.so](http://state.so) 缺失 + stale lock**

gb200-5 容器里缺 reset_kv_[state.so](http://state.so)，而且残留 build/lock，DP0-3 在 finish cleanup 时卡住。删 lock + 预编译 so 后解决。后来在 `kv_offload/utils.py::load_kernel_module()` 里加了 stale lock 自动清理（默认 600 秒超时）。

**compact_cache_ids_block 类型不匹配**

CUDA kernel 用 data_ptr<int>()（int32），但 PyTorch 默认 tensor 是 int64。需要显式 `.to(torch.int32).contiguous()`。

**dp_affinity_scheduler round-robin 与 recv_requests broadcast 不匹配**

这是 576 token 边界的根因——其实和 input length 无关。D-only decode 下，recv_requests 只有 global_rank==0 从 ZMQ 读请求，其他 rank 通过 broadcast 接收。但 dp_affinity_scheduler 的 round_robin 会把请求发给 DP1/2/3，这些 rank 不读 ZMQ，消息被静默丢弃。4 个 worker 只有 1 个能收到，成功率约 25%。

修复：kv_transfer_params 为 None 时始终发 workers[0]。

**TpModelWorkerClient.get_memory_pool() 返回值分叉**

普通 worker 在 decode+kv offload 下返回 3 个 pool（含 sparse allocator），但 overlap client 只返回 2 个。scheduler 解包时期望 3 个，直接 ValueError。修复为委托 `self.worker.get_memory_pool()`。

**trans_stream 漏传**

graph capture 路径里 forward_batch 没带 trans_stream。补了 `forward_batch.trans_stream = self.trans_stream` 和 `copy_forward_batch_common_attributes()` 的 trans_stream 传递。

**offload 解包错位**

graph/overlap decode 支线按旧的 4 元返回值解包，但 offload 路径返回 8 元组。改为直接透传完整元组。

**SET_DEEP_DP_MAX_TOKENS 默认值太小**

默认 32，但 graph capture 过程中 forward_decode 会到 batch_size=72。触发 deep_ep.cpp:1262 的 `num_max_dispatch_tokens_per_rank` 断言。改成 256 后解决。

**offload IPC 后缀不一致**

gb200-6 上 controller 连接 encoder_ipc_name_4..7，但 encoder 绑定 encoder_ipc_name_0..3。修复：offload 模式下统一使用 node-local DP rank。

**request/cache status EP 全组同步误用**

TP1/DP8/EP-MoE 下，每个 DP scheduler 都是 tp_rank=0，不需要 EP 组 broadcast。只有 enable_dp_attention 才需要。误用 EP 组会导致 reading_cache_queue 长度不一致，dist.all_gather 卡死。

## 4.4 Intern_FlashMLA stage1 illegal memory access

这是当前仍未解决的 blocker。现象是 offload + dual_attention 组合下，第一条 decode 请求在 stage1 FlashMLA sparse-fp8 decode kernel 触发 illegal memory access：

```
/opt/aiak/Intern_FlashMLA/csrc/sm100/decode/sparse_fp8/splitkv_mla.cu:766
CUDA error: an illegal memory access was encountered
```

关键信息：

- 输入摘要：q_shape=(1,1,128,576)，kv_cache_shape=(1171,64,1,656)，cache_seqlens=[2]
- hitted_indices 有效值范围 [64,65]，没有越界
- missed_count=0
- 控制组：dual_attention + no offload + no graph 返回 200，说明 dual_attention 本身能工作
- 崩溃发生在 fast_intra_layer_h2d 之前，还没执行到 H2D copy

所以问题出在 offload 引入的 hit/miss 稀疏输入和 stage1 FlashMLA 的交互上。具体是 splitkv_[mla.cu](http://mla.cu) 对小 batch/短序列 + offload 稀疏索引的假设有问题，还是传入的某个参数实际值不对，还没定位到。

## 4.5 Mooncake 通信路径

Mooncake 在 GB200 上的问题是：源码只有 rdma 和 tcp 两个 backend，没有 NVLink/IMEX/MNNVL。

rdma 路径：默认 nic_priority_matrix 写的是 mlx5_bond_0，GB200 上不存在。改成 mlx5_0,mlx5_1,mlx5_4,mlx5_5 后 transport 能 install，但 registerRecvBuffer 报 Bad address。这说明传统 RDMA 在 GB200 上不直接可用。

tcp 路径：设 `MOONCAKE_USE_RDMA=FALSE` 后 Mooncake 走 TCP，P/D 可以各自启动。但 pdtest mock_ic 验证时，P TP0 在 send_tensors 后触发 Fatal Python error: Segmentation fault，栈顶落在 [libucs.so](http://libucs.so).0。D 端 30 秒后报 Wait for transfer msg failed。

这不是 P/D 串通的代码逻辑问题，而是 Mooncake 底层 UCX 库的 crash。当前真源脚本已固定 MOONCAKE_USE_RDMA=FALSE。

## 4.6 其他代码修复

**/v1/chat/completions 恢复**

原来被硬编码成 501 Not Implemented。pdtest 硬编码用这个 endpoint，必须恢复。

**DeepSeek-V3.2 无 chat_template 的 fallback**

tokenizer 没有 chat_template，直接调用 apply_chat_template 会 ValueError。[adapter.py](http://adapter.py) 和 [encoder.py](http://encoder.py) 都要补 fallback：无 chat_template 时直接拼接 message content 做 tokenize。只补一边不够，两边都补了才让 pdtest 不再 500。

**parallel_[state.py](http://state.py) MNNVL 补丁**

deep_ep Buffer 两处调用加 `allow_mnnvl=True, use_fabric=True`，这是跨节点 MNNVL 通信的必要条件。

**memory_[pool.py](http://pool.py) tensor view 对齐**

NSATokenToKVPoolOffload 的 tensor view 按 page_size 对齐，不修的话 shape 会和后续 kernel 假设不一致。

---

# 5 当前状态

截至 4 月 22 日：

| 配置组合 | 状态 |
| --- | --- |
| P 节点 gb200-5+6 | 已跑通，curl 返回正常 |
| P 节点 gb200-1+0 | 已跑通，需 NCCL_IB_HCA + imex |
| D-only no-graph + offload + no dual | 最小请求返回 200 |
| D-only graph + offload + no dual | 最小请求返回 200（需 SET_DEEP_DP_MAX_TOKENS=256） |
| D-only graph + offload + dual | 启动后首请求 crash，stage1 FlashMLA illegal memory access |
| 手工两段式 P→D | 已打通（AIAK_DEBUG 模式） |
| pdtest mock_ic | P 返回首 token，D enqueue 成功，P send_tensors 后 UCX segfault |
| 端到端自动 PD | 未完成 |

当前两个前沿 blocker：

1. offload + dual_attention 的 stage1 Intern_FlashMLA illegal memory access
2. Mooncake send_tensors 后 UCX/libucs segfault

---

# 6 经验总结

**先查环境再调代码。** gb200-5/6 第一次就跑通不是因为代码没问题，而是三件事恰好都对。换机器后先跑前置检查清单，比试错快得多。

**GB200 不是大号 HGX。** Fabric Manager 在 switch tray 上不在 compute tray 上，MNNVL 不是传统 RDMA，NCCL 可能选中只有 link-local GID 的 NIC。这些认知差异会导致排查方向完全走偏。

**假设所有机器环境一致是最危险的假设。** 同批交付的 14 台机器，驱动版本、IB 状态、imex 运行状态都不一样。不显式检查就换机器，只会陷入"明明一样的代码为什么不行"的循环。

**AI 交互式调试效率不低但有代价。** 没有 K8S 和自动化，全靠 SSH + AI 辅助看日志定位问题，两周推进到这个程度算快。但上下文管理成本高——每次启动都要重新对齐环境状态、代码版本、历史排查结论。本地 timeline 文档和 Notion 记录是唯一不会丢的上下文。

**JIT 编译的 lock 文件是个定时炸弹。** 多个 DP rank 并发触发 JIT 编译时，某个 rank 的失败构建会留下 stale lock，把同 build 目录下其他 op 的编译也卡住。加了超时清理机制后好了一些，但根本解法是预编译或随镜像分发 .so。