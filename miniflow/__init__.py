"""miniflow — DeepInfer 的教学复刻版。

模块划分与 DeepInfer (deepseek-ai/DeepInfer) 一一对应：

    miniflow/                deepinfer/
    ---------------------    ---------------------------------
    config.py            <-> config.py / environ.py
    sequence.py          <-> sequence.py
    sampling_params.py   <-> sampling_params.py
    tokenizer.py         <-> deepseek-recipe (Rust 编解码层思想)
    tensor.py            <-> FlashMLA/DeepGEMM 等算子层 + csrc/
    model/               <-> model_executor/layers/
    cache/paged_kv.py    <-> KV cache (PagedAttention 思想)
    cache/prefix_cache.py<-> KVCache Storage (3FS) 前缀复用思想
    scheduler.py         <-> engine/ 调度器 (continuous batching + 异步调度)
    sampler.py           <-> 批次不变推理 / 确定性采样
    speculative.py       <-> DSpark 推测解码
    structured_output.py <-> structure_output/
    engine.py            <-> engine/ LLMEngine
    entrypoints/cli.py   <-> entrypoints/
"""

__version__ = "0.1.0"