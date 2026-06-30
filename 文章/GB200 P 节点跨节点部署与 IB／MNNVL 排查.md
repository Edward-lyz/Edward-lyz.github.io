# GB200 P 节点跨节点部署与 IB/MNNVL 排查

状态: 已完成
截止日期: 04/16/2026
任务类型: FEATURE
小结: P 节点在 gb200-5+6 和 gb200-1+0 上均已跑通。根因是 IB GID 不匹配、nvidia-imex 未运行、驱动版本不一致，三项分别修复后可稳定部署
描述: 4/14: 首次跨节点部署 → 4/15: IB拓扑排查 → 4/16: MNNVL实测+P节点forward修复 → 4/17: 多机器对测试+GID根因+gb200-1+0成功

## 任务描述

在 GB200 NVL72 集群上部署 P 节点（Prefill），解决跨节点 NCCL/MNNVL 通信问题，使 P 节点可正常推理。

## 子任务

- [x]  跨节点 TP=8 NCCL 初始化（32 channels P2P）
- [x]  DeepEP hybrid-ep 单节点 4 GPU 测试通过
- [x]  IB 拓扑排查：gb200-7 IB PORT_DOWN，gb200-9↔gb200-12 不通
- [x]  MNNVL 启用条件实测（2MPI×4GPU/8MPI×1GPU 均 MNNVL=1）
- [x]  P 节点 forward 链路修复（tokenizer_manager/apply_rope/fast_hadamard_transform）
- [x]  P 节点 curl max_tokens=1 返回正常
- [x]  多机器对 NCCL 跨节点测试，定位 IB GID 不匹配根因
- [x]  NCCL_IB_HCA 修复 + gb200-1+gb200-0 P 节点部署成功
- [x]  NCCL_SOCKET_IFNAME + NCCL_SOCKET_FAMILY 修复
- [x]  USE_CUTLASS_FP8_BLOCK_GEMM=0 修复 block-fp8 linear

## 支持文件

P 真源脚本：PRIVATE/scripts/start_gb200_aiak_p_prefill_graph_[offload.sh](http://offload.sh)