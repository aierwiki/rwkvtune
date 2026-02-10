# RWKVTune Inference Module

RWKVTune inference module, providing inference support for training processes (e.g., GRPO).

## Module Architecture

```
                    ┌─────────────────────────────────────────────────────┐
                    │              Unified Low-level Scheduler             │
                    │              SchedulerCore                           │
                    │  ┌─────────────────────────────────────────────┐    │
                    │  │ • Slot Management                              │    │
                    │  │ • State Pool (GPU)                            │    │
                    │  │ • Prefill/Decode Scheduling                  │    │
                    │  │ • Chunked Prefill                            │    │
                    │  │ • State Cache Integration                    │    │
                    │  └─────────────────────────────────────────────┘    │
                    └─────────────────────────────────────────────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              │                           │                           │
              ▼                           ▼                           ▼
    ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
    │  BatchGenerator │         │   RWKVEngine    │         │  InferencePipeline│
    │  (Sync Batching)│         │  (Async Service)│         │  (High-level API) │
    ├─────────────────┤         ├─────────────────┤         ├─────────────────┤
    │ • GRPO Training │         │ • API Service   │         │ • Single Generate│
    │ • LLM.generate()│         │ • Streaming     │         │ • Batch Generate │
    │ • Offline Batch │         │ • State Cache   │         │ • Preset Modes   │
    └─────────────────┘         └─────────────────┘         └─────────────────┘
```

## Directory Structure

```
rwkvtune/
├── inference/                  # Inference core module
│   ├── __init__.py
│   ├── scheduler_core.py       # 🆕 Unified scheduler core (shared backend)
│   ├── core.py                 # TokenSampler, StateManager, GenerationCore
│   ├── state_cache.py          # State Cache system (Trie prefix matching)
│   ├── batch_generator.py      # Batch inference (uses SchedulerCore)
│   ├── pipeline.py             # InferencePipeline (high-level interface)
│   ├── generator.py            # TextGenerator (preset config)
│   └── structured_output.py    # JSON Schema constrained generation
├── engine/                     # Async engine
│   ├── core.py                 # RWKVEngine (uses SchedulerCore)
│   ├── scheduler.py            # Request scheduler
│   └── sequence.py             # Sequence state management
├── entrypoints/                # vLLM-style entry points
│   └── llm.py                  # LLM class
├── sampling_params.py          # SamplingParams parameter class
└── outputs.py                  # RequestOutput, CompletionOutput
```

---

## Quick Start

### 1. vLLM-style Batch Inference (Recommended)

```python
from rwkvtune import LLM, SamplingParams

# Load model (without State Cache)
llm = LLM(
    model="/path/to/model",
    device="cuda",
    dtype="bf16",
    max_batch_size=256,       # Max concurrent sequences
    max_batch_tokens=8192,    # Max tokens per Forward
)

# Set sampling parameters
params = SamplingParams(
    max_tokens=256,
    temperature=0.8,
    top_p=0.95,
)

# Batch generation
prompts = ["Hello, my name is", "The future of AI is"]
outputs = llm.generate(prompts, params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
```

### 1.1 Enable State Cache (Repeated Prompt Scenarios)

```python
from rwkvtune import LLM, SamplingParams

# Load model (with State Cache)
llm = LLM(
    model="/path/to/model",
    device="cuda",
    dtype="bf16",
    use_state_cache=True,         # Enable State Cache
    state_cache_memory_gb=4.0,    # Cache memory limit (GB)
)

# First inference: cache miss, normal computation
prompts = ["System prompt..." + "User: Hello"] * 10
outputs1 = llm.generate(prompts, SamplingParams(max_tokens=50))

# Second inference: cache hit, skip Prefill, 3-5x speedup
outputs2 = llm.generate(prompts, SamplingParams(max_tokens=50))

# View cache statistics
if llm.state_cache:
    stats = llm.state_cache.get_stats()
    print(f"Cache hit rate: {stats['hit_rate']:.1%}")
    print(f"Cache entries: {stats['entry_count']}")
```

### 2. High-level Pipeline Interface

```python
from rwkvtune import InferencePipeline

# Create inference pipeline
pipeline = InferencePipeline(
    model_path="/path/to/model",
    model_config="7.2b",
    device="cuda",
    precision="bf16",
)

# Single generation
response = pipeline.generate(
    prompt="Hello, how are you?",
    max_tokens=100,
    temperature=0.7,
)

# Batch generation
responses = pipeline.generate(
    prompt=["Hello", "World"],
    max_tokens=100,
)
```

### 3. SchedulerCore Low-level Interface (Advanced)

```python
from rwkvtune import AutoModel, AutoTokenizer
from rwkvtune.inference import SchedulerCore, SequenceState, SeqStatus

# Load model
model = AutoModel.from_pretrained("/path/to/model", device="cuda")
tokenizer = AutoTokenizer.from_pretrained("/path/to/model")

# Create SchedulerCore
scheduler = SchedulerCore(
    model=model,
    tokenizer=tokenizer,
    max_batch_size=64,
    max_batch_tokens=8192,
    prefill_chunk_size=512,
)

# Prepare sequences
sequences = [
    SequenceState(
        seq_id=0,
        prompt_tokens=tokenizer.encode("Hello"),
        max_new_tokens=100,
        temperature=0.8,
        eos_token_id=tokenizer.eos_token_id,
    ),
    SequenceState(
        seq_id=1,
        prompt_tokens=tokenizer.encode("World"),
        max_new_tokens=100,
        temperature=0.8,
        eos_token_id=tokenizer.eos_token_id,
    ),
]

# Add to scheduler
scheduler.add_sequences(sequences)

# Run to completion
all_tokens = {0: [], 1: []}
for step_tokens in scheduler.run_to_completion_sync():
    for seq_id, token_id in step_tokens:
        all_tokens[seq_id].append(token_id)

# Decode results
for seq_id, tokens in all_tokens.items():
    print(f"Seq {seq_id}: {tokenizer.decode(tokens)}")
```

---

## Core Components Details

## 1. SchedulerCore (Unified Scheduler) 🆕

`SchedulerCore` is the shared low-level scheduler for `BatchGenerator` and `RWKVEngine`, implementing the core logic of Continuous Batching.

### Core Features

- **Slot Management**: GPU physical state slot allocation/release
- **State Pool**: Pre-allocated GPU state tensor pool
- **Continuous Batching**: Decode-first + Prefill-filling scheduling strategy
- **Chunked Prefill**: Chunked processing of long prompts
- **State Cache Integration**: Optional state cache support

### Architecture Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           SchedulerCore                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                     State Pool (GPU)                             │   │
│  │  ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐  │   │
│  │  │ Slot 0  │ Slot 1  │ Slot 2  │ Slot 3  │   ...   │ Slot N  │  │   │
│  │  │  Seq A  │  Seq B  │ (empty) │  Seq D  │         │ (empty) │  │   │
│  │  └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘  │   │
│  │  • att_x_prev: [n_layer, max_batch, n_embd]                      │   │
│  │  • att_kv:     [n_layer, max_batch, n_head, head_size, head_size]│   │
│  │  • ffn_x_prev: [n_layer, max_batch, n_embd]                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐       │
│  │  waiting_queue  │ → │ prefilling_seqs │ → │  running_seqs   │       │
│  │  Waiting        │   │  Prefilling     │   │  Decoding       │       │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_batch_size` | int | 64 | Max concurrent sequences (Slot count) |
| `max_batch_tokens` | int | 8192 | Max tokens per Forward |
| `prefill_chunk_size` | int | 512 | Chunked Prefill block size |
| `state_cache` | StateCache | None | State Cache instance (optional) |

### Workflow

```
Step 1: Schedule (_schedule)
  ├── Allocate Slot from waiting_queue
  ├── Schedule Decode (higher priority, 1 token per sequence)
  └── Schedule Prefill (use remaining budget, chunk by chunk_size)

Step 2: Execute Decode (_execute_decode)
  ├── Gather states from state_pool
  ├── Forward: [B, 1] → [B, 1, C]
  ├── Scatter states back
  └── Sample next token

Step 3: Execute Prefill (_execute_prefill)
  ├── Group by chunk length (avoid padding)
  ├── Gather states
  ├── Forward: [B, chunk_len] → [B, chunk_len, C]
  ├── Scatter states back
  └── If prefill done, sample first token
```

---

## 2. State Cache System

State Cache is a state caching system designed for RWKV models, accelerating requests with identical or similar prefixes by caching inference intermediate states.

### Core Features

- **Trie Index**: Efficient prefix matching
- **Three checkpoint types**: `system_prompt` / `interval` / `full_prompt`
- **Unified LRU eviction**: Smart memory management
- **CPU Storage**: Cache stored in CPU memory, saving GPU memory
- **Auto configuration**: Optimal parameters from memory limit
- **LLM Integration**: One-click enable via `use_state_cache=True`

### Performance Benchmark Results

| Scenario | Prompt Length | No Cache | With Cache (subsequent) | **Speedup** |
|----------|---------------|----------|-------------------------|-------------|
| Identical prompt | 365 tokens | 0.42s | 0.13s | **3.3x** |
| Data labeling scenario | 367 tokens | 1.06s | 0.22s | **4.8x** |

| Metric | No Cache | With Cache (hit) | **Improvement** |
|--------|----------|------------------|-----------------|
| Prefill speed | ~3,300 tokens/s | ~365,000 tokens/s | **110x** |
| Overall throughput | 4.3 prompts/s | 5.0 prompts/s | **1.15x** |

> **Note**: Decode phase time is unchanged; overall speedup depends on Prefill proportion. Longer prompts and shorter outputs yield more noticeable speedup.

### Cache Levels

| Level | Description | Hit Condition | Use Case |
|-------|-------------|---------------|----------|
| `NONE` | Disabled | Never hits | Debugging, testing |
| `EXACT` | Exact match | Identical prompts | Many repeated requests |
| `PREFIX` | Prefix match (recommended) | Shared prefix | General use, multi-turn dialogue |

### Usage

#### Method 1: Enable via LLM Class (Recommended)

```python
from rwkvtune import LLM, SamplingParams

# Enable State Cache
llm = LLM(
    model="/path/to/model",
    use_state_cache=True,         # Enable cache
    state_cache_memory_gb=4.0,    # Memory limit
)

# Normal usage, cache works automatically
outputs = llm.generate(prompts, SamplingParams(max_tokens=50))

# View statistics
stats = llm.state_cache.get_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
```

#### Method 2: Create StateCache Manually

```python
from rwkvtune.inference import CacheConfig, CacheLevel, StateCache

config = CacheConfig(
    max_cache_memory_gb=4.0,         # Max cache memory (GB)
    max_cache_entries=None,         # Max entries (auto-calculated)
    cache_level=CacheLevel.PREFIX,   # NONE/EXACT/PREFIX
    cache_first_message=True,       # Whether to cache first message
)

cache = StateCache(model, config=config)
```

### Workflow

```
Request 1: [System Prompt][User: Hello]
        ↓ First request, full Prefill computation
        ↓ Cache Full Prompt checkpoint to CPU memory

Request 2: [System Prompt][User: Hello]  (same prompt)
        ↓ Lookup Trie, exact hit
        ↓ Load state from CPU to GPU Slot
        ↓ Skip Prefill, go directly to Decode
        ↓ Effect: Prefill time ~0
```

### Recommended Scenarios

| Scenario | Recommended | Reason |
|----------|-------------|--------|
| Same prompt multiple inferences | ✅ Highly recommended | 100% cache hit |
| Data labeling (shared system prompt) | ✅ Highly recommended | System prompt reusable |
| Batch processing similar requests | ✅ Recommended | Prefix matching effective |
| Every prompt different | ❌ Not recommended | Extra overhead |

---

## 3. Continuous Batching

Continuous Batching is the core technique for high-throughput inference, implemented uniformly by `SchedulerCore`.

### Scheduling Strategy

```
Each Step:
┌─────────────────────────────────────────────────────────────────┐
│ Token Budget = 8192                                              │
│                                                                  │
│ 1. Decode scheduling (higher priority)                           │
│    • Each running sequence consumes 1 token                      │
│    • Ensures low latency                                        │
│                                                                  │
│ 2. Prefill scheduling (use remaining budget)                    │
│    • Chunk by chunk_size                                        │
│    • Short requests can start generating sooner                  │
└─────────────────────────────────────────────────────────────────┘
```

### Usage Example

```python
# BatchGenerator (sync, offline batch)
from rwkvtune.inference import BatchGenerator, GenerationConfig

generator = BatchGenerator(model, tokenizer, batch_strategy="continuous")
config = GenerationConfig(
    max_length=256,
    max_batch_tokens=8192,
    chunk_size=512,
)
result = generator.generate(prompts=["Hello", "World"], generation_config=config)
```

```python
# RWKVEngine (async, API service)
from rwkvtune.engine import RWKVEngine

engine = RWKVEngine(
    model=model,
    tokenizer=tokenizer,
    max_batch_size=16,
    max_prefill_tokens=4096,
    prefill_chunk_size=512,
)

output_queue = await engine.add_request("req-1", "Hello", {"max_tokens": 100})
async for token_id in output_queue:
    if token_id is None:
        break
    print(tokenizer.decode([token_id]), end="")
```

---

## 4. Chunked Prefill

Chunk-based processing of long prompts to control memory peak.

### Problem

```
Prompt: 2000 tokens
One-shot: Need [B, 2000, C] intermediate tensors → High memory peak
```

### Solution

```
Original: [t1, t2, ..., t2000]
        ↓ chunk_size=512
Chunks: [t1..t512] → [t513..t1024] → [t1025..t1536] → [t1537..t2000]
         │              │               │              │
         ▼              ▼               ▼              ▼
      state_1  →    state_2    →    state_3    →  state_final
```

### Memory Comparison

| Prompt Length | One-shot | Chunked (512) |
|---------------|----------|---------------|
| 1000 tokens | ~4GB peak | ~1GB peak |
| 2000 tokens | ~8GB peak | ~1GB peak |
| 4000 tokens | OOM | ~1GB peak |

---

## 5. Sampling Strategy

### TokenSampler (inference/core.py)

```python
from rwkvtune.inference.core import TokenSampler

# Single sample
token_id, log_prob = TokenSampler.sample(
    logits,                    # [vocab_size]
    temperature=0.8,           # Temperature (0 = greedy)
    top_p=0.95,                # Nucleus sampling
    top_k=50,                  # Top-k sampling
    token_ban=[1, 2, 3],       # Banned tokens
)

# Batch sample (GPU optimized)
tokens, log_probs = TokenSampler.sample_batch(
    logits_batch,              # [B, vocab_size]
    temperature=0.8,
    top_p=0.95,
)
```

### Sampling Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 1.0 | Temperature; higher = more random, 0 = greedy |
| `top_p` | float | 1.0 | Nucleus sampling; keep tokens with cum prob <= top_p |
| `top_k` | int | 0 | Top-k sampling; 0 = no limit |
| `token_ban` | List[int] | [] | List of banned token IDs |

---

## 6. Structured Output

JSON Schema constrained generation using the `lm-format-enforcer` library.

```python
from rwkvtune.inference import JsonSchemaEnforcer

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"]
}

enforcer = JsonSchemaEnforcer(tokenizer, schema)

for step in range(max_tokens):
    masked_logits = enforcer.mask_logits(logits)
    token_id, _ = TokenSampler.sample(masked_logits, temperature=0.8)
    is_done = enforcer.update(token_id)
    if is_done:
        break
```

---

## 7. vLLM-style API

### LLM Class

```python
from rwkvtune import LLM, SamplingParams

llm = LLM(
    model="/path/to/model",
    device="cuda",
    dtype="bf16",
    gpu_id=0,
    max_batch_size=256,           # Max concurrent sequences
    max_batch_tokens=8192,       # Max tokens per Forward
    prefill_chunk_size=512,      # Chunked Prefill block size
    use_state_cache=False,       # Enable State Cache
    state_cache_memory_gb=4.0,   # State Cache memory limit (GB)
)

params = SamplingParams(
    max_tokens=256,
    temperature=0.8,
    top_p=0.95,
    top_k=0,
    stop=["</s>"],
    stop_token_ids=[0],
)

outputs = llm.generate(prompts, params)
```

### LLM Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | - | Model path |
| `device` | str | "cuda" | Inference device |
| `dtype` | str | "bf16" | Model precision |
| `gpu_id` | int | None | GPU ID |
| `max_batch_size` | int | 256 | Max concurrent sequences |
| `max_batch_tokens` | int | 8192 | Max tokens per Forward |
| `prefill_chunk_size` | int | 512 | Chunked Prefill block size |
| `use_state_cache` | bool | False | Enable State Cache |
| `state_cache_memory_gb` | float | 4.0 | State Cache memory limit (GB) |

### SamplingParams

```python
from rwkvtune import SamplingParams

params = SamplingParams(
    max_tokens=256,          # Max generated tokens
    temperature=1.0,         # Temperature (0 = greedy)
    top_p=1.0,               # Nucleus sampling
    top_k=0,                 # Top-k sampling
    stop=None,               # Stop string list
    stop_token_ids=None,     # Stop token IDs
    skip_special_tokens=True,# Skip special tokens
    token_ban=None,          # Banned tokens
)
```

---

## Configuration Parameter Summary

### SchedulerCore (Low-level Scheduling)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_batch_size` | 64 | Max concurrent sequences |
| `max_batch_tokens` | 8192 | Token budget per Forward |
| `prefill_chunk_size` | 512 | Chunked Prefill block size |
| `state_cache` | None | State Cache instance |

### GenerationConfig (BatchGenerator)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_length` | 256 | Max generation length |
| `temperature` | 1.0 | Sampling temperature |
| `top_p` | 1.0 | Nucleus sampling |
| `top_k` | 0 | Top-k sampling |
| `max_batch_tokens` | 8192 | Token budget per Forward |
| `chunk_size` | 512 | Chunked Prefill block size |

### RWKVEngine (Async Service)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_batch_size` | 16 | Max concurrent sequences |
| `max_prefill_tokens` | 4096 | Max tokens per prefill |
| `prefill_chunk_size` | 512 | Chunked Prefill block size |
| `state_cache` | None | State Cache instance |

---

## Performance Tuning Best Practices

### 1. Memory Configuration

```python
cache = create_state_cache(
    model=model,
    max_cache_memory_gb=4.0,  # Adjust based on CPU memory
)
```

### 2. Batch Configuration

```python
scheduler = SchedulerCore(
    model=model,
    tokenizer=tokenizer,
    max_batch_size=64,        # Adjust based on GPU memory
    prefill_chunk_size=512,  # Control memory peak
)
```

### 3. vLLM-style Large Batch Processing

```python
llm = LLM(
    model="/path/to/model",
    max_num_seqs=64,  # Auto batching to avoid OOM
)

# Process 10000 prompts
outputs = llm.generate(prompts, params, use_tqdm=True)
```

---

## Testing

```bash
# Run State Cache performance test
CUDA_VISIBLE_DEVICES=6 python tests/test_state_cache_detailed_benchmark.py

# Run State Cache unit tests
CUDA_VISIBLE_DEVICES=7 pytest tests/test_inference/test_state_cache.py -v

# Run batch inference test
CUDA_VISIBLE_DEVICES=7 pytest tests/test_inference/test_batch_generator.py -v

# Quick State Cache verification
CUDA_VISIBLE_DEVICES=6 python test_cache_simple.py
```

### State Cache Performance Test Output Example

```
Scenario: Data labeling (10 prompts) - with cache
Prompt count: 10
Avg prompt tokens: 366.6

  Run 1: 2.561s (cache hits: 0/10)
  Run 2: 0.222s (cache hits: 10/10) ← Cache active, 4.8x speedup
  Run 3: 0.222s (cache hits: 10/10)
```

---

## API Service

### Start Service

```bash
python -m rwkvtune.cli.serve \
    --model-path /path/to/model \
    --host 0.0.0.0 \
    --port 8000 \
    --max-batch-size 16 \
    --prefill-chunk-size 512
```

### Cache Monitoring

```bash
# Get cache statistics
curl http://localhost:8000/cache/stats

# Clear cache
curl -X POST http://localhost:8000/cache/clear
```

---

**Maintainer**: RWKVTune Team  
**Last Updated**: 2026-02-06

### Changelog

- **2025-12-05**: LLM class integrated with State Cache; added `use_state_cache` and `state_cache_memory_gb` parameters
- **2025-12-04**: Implemented Continuous Batching; unified SchedulerCore low-level scheduling
