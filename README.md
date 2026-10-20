# deepinfer-lab — DeepInfer 原理实验室

> 解读 [DeepInfer](https://github.com/deepseek-ai/DeepInfer)（DeepSeek 于 2026-10-20 开源的生产级大模型推理框架）的底层原理与设计哲学，并用 Python + C++ 从零复刻一个教学版推理引擎 **miniflow**。

## 为什么做这个项目

大模型的能力边界由模型本身划定，但如何以合理成本把模型服务到海量用户，取决于一个长期不被公众了解的中间层——**推理框架**。DeepInfer 是支撑 DeepSeek V4.1 Flash 日均 155 万亿 token 服务的生产系统，它把「kernel、通信、调度、缓存管理」编织成一个集群级系统。

这个仓库做两件事：

1. **深度解读**：基于官方仓库与《Inference Deployment Q&A》，逐模块拆解 DeepInfer 的架构与设计取舍（见文档站）。
2. **从零复刻**：用 DeepInfer 的原生技术栈（Python 引擎 + C++ 算子）实现一个 CPU 上可跑的教学引擎 `miniflow`，每一行代码对应 DeepInfer 的一个生产组件。

## miniflow 与 DeepInfer 的模块映射

| DeepInfer (生产) | miniflow (教学) | 对应原理 |
| --- | --- | --- |
| `deepinfer/config.py` | `miniflow/config.py` | 引擎配置与环境开关（`DEEPINFER_*` 环境变量体系） |
| `deepinfer/sequence.py` / `sampling_params.py` | `miniflow/sequence.py` / `sampling_params.py` | 请求抽象、采样参数、`seq_seed` 确定性采样 |
| deepseek-recipe（Rust 编解码层） | `miniflow/tokenizer.py` | BPE 分词、特殊 token 防注入 |
| FlashMLA / DeepGEMM / TileLang kernels | `miniflow/tensor.py` + `csrc/` (pybind11) | 算子层：NumPy 参考实现 + C++ 加速版 |
| `model_executor/layers/` | `miniflow/model/` | RMSNorm / RoPE / SWA 注意力 / MLA 思想 / MoE 路由 |
| KVCache Storage (3FS) + Paged KV | `miniflow/cache/paged_kv.py` / `prefix_cache.py` | 分页 KV、块表、哈希链前缀缓存 |
| `engine/` 异步调度 | `miniflow/scheduler.py` | Continuous batching、CPU/GPU 流水线重叠 |
| DSpark 推测解码 | `miniflow/speculative.py` | Draft 模型 + 置信度调度验证（变长验证） |
| Batch-invariant / 确定性采样 | `miniflow/sampler.py` | 批次不变推理、逐请求独立采样 |
| `structure_output/` | `miniflow/structured_output.py` | 语法约束 → 采样掩码 |
| `entrypoints/` | `miniflow/entrypoints/cli.py` | 离线推理入口 |
| EPLB 专家负载均衡 | `miniflow/model/moe.py` | 冗余专家 + 统计窗口重分配（演示） |

## 快速开始

```bash
# 1. 训练一个微型字符级语言模型（NumPy 实现，CPU 数秒~数分钟）
python tools/train.py --out checkpoints/tiny.npz

# 2. 离线推理（对应 DeepInfer 的 examples/offline_inference.py）
python examples/offline_inference.py

# 3. 各核心机制演示
python examples/demo_paged_kv.py        # 分页 KV cache 与内存碎片
python examples/demo_prefix_cache.py    # 前缀缓存命中如何省掉重复 prefill
python examples/demo_scheduler.py       # continuous batching 吞吐对比
python examples/demo_speculative.py     # 推测解码加速比
python examples/demo_structured.py      # 结构化输出（JSON 约束）

# 4. 测试
python -m pytest tests/ -q

# 5. 文档站（MkDocs Material）
pip install -r requirements.txt
mkdocs serve    # http://127.0.0.1:8000
```

## 文档站目录（深度解读）

- **总览**：DeepInfer 是什么、三大设计目标、E/P/D 分离架构
- **内核层**：FlashMLA / DeepGEMM / DeepEP / TileLang 与低精度（FP4）
- **并行策略**：Attention DP、Wide-EP、EPLB 专家负载均衡
- **KV Cache**：分页管理、3FS 分布式存储、SWA、有界重放（Bounded Replay）
- **调度**：Continuous Batching、异步零开销调度器、四级 Prefill 负载均衡
- **推测解码**：DSpark、置信度头、变长验证
- **采样**：确定性采样与批次不变推理
- **结构化输出**：语法自动机与采样掩码
- **deepseek-recipe**：Rust 编解码层的工程哲学
- **设计哲学总结**：十条可迁移的设计原则

## 参考资料与致谢

- [deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer)（Apache-2.0）
- [DeepSeek V4.1 Flash Inference Deployment Q&A](https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md)
- [deepseek-ai/deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe)
- DeepInfer 建立在 [vLLM](https://github.com/vllm-project/vllm) 之上，并借鉴了 [SGLang](https://github.com/sgl-project/sglang) 与 [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

## License

MIT（本仓库教学代码）。解读内容基于 Apache-2.0 的 DeepInfer 公开资料整理。