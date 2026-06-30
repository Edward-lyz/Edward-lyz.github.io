# GB200 D 节点部署与 Offload/Graph 推进

状态: 已完成
截止日期: 04/21/2026
任务类型: FEATURE
小结: D-only graph+offload+no-dual 已可用，最小请求返回200。offload+dual_attention 的 stage1 FlashMLA sparse-fp8 illegal memory access 仍未解决
描述: 4/17: D节点首次启动 → 4/18: 手工两段式打通 → 4/20: D-only mock KV → 4/20: offload根因缩到FlashMLA → 4/20: graph+offload+no-dual通过 → 4/21: full config仍卡在dual+offload

## 任务描述

在 GB200 NVL72 上部署 D 节点（Decode），从 no-graph 到 graph，从 no-offload 到 offload，逐步推进 D-only decode 路径。

## 子任务

- [x]  D 节点 no-graph + official flashmla_decode 启动到 Uvicorn
- [x]  手工两段式 P→D 已打通（AIAK_DEBUG 模式）
- [x]  D-only mock KV 路径确认（CACHE_MOCK=random/mock）
- [x]  D 拓扑纠偏为 TP1/DP8
- [x]  offload + no dual_attention: 最小请求返回 200（reset_kv_[state.so](http://state.so) 预编译）
- [x]  graph + offload + no dual_attention: SET_DEEP_DP_MAX_TOKENS=256 后起服
- [x]  graph + offload + no dual_attention: 最小请求返回 200
- [ ]  offload + dual_attention: stage1 Intern_FlashMLA illegal memory access 未解决
- [ ]  graph + offload + dual_attention: graph capture 阶段 fast_intra_layer_h2d 路径 crash

## 支持文件

D 真源脚本：PRIVATE/scripts/start_gb200_aiak_d_decode_graph_[offload.sh](http://offload.sh)