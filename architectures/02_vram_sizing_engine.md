# VRAM Sizing Engine

Before any model touches the GPU, the system answers: **"Will this model fit, and at what sequence length?"** This happens **statically** — no weights are downloaded.

The core logic lives in:
- `runtime_common/model_static_check.py` — the math engine
- `generate_gpu_config.py` — text model config generator
- `generate_gpu_vision_config.py` — vision model config generator

---

## The Memory Model

```
Total VRAM Budget  =  target_vram_gb × 1024 MB
                   ↓
┌─────────────────────────────────────────────────────────────┐
│  weights_mb     ←  model parameters (checkpoint size or     │
│                    estimated from config.json dims)          │
│  + kv_cache_mb  ←  2 × layers × seq_len × kv_heads ×       │
│                    head_dim × dtype_bytes                    │
│  + overhead_mb  ←  20% reserve + runtime fragmentation       │
└─────────────────────────────────────────────────────────────┘
                         ↓
              classify_fit(total_mb, vram_gb)
```

### KV Cache Formula

```python
kv_cache_mb = (
    2 * num_layers * seq_len * num_kv_heads * head_dim * dtype_bytes
) / (1024 ** 2)
```

Where:
- `2` = key + value tensors
- `dtype_bytes` = 2 for fp16/bf16, 4 for fp32
- `num_kv_heads` = may be fewer than `num_attention_heads` (GQA/MQA)

---

## Weight Estimation

### Method 1: Real Checkpoint Size (Preferred)

The generator fetches `model.safetensors.index.json` or `pytorch_model.bin.index.json` from HuggingFace and reads the `metadata.total_size` field. This is exact.

### Method 2: Config Heuristic (Fallback)

If the checkpoint index is unavailable, weights are estimated from `config.json` dimensions:

```python
# Attention: Q, K, V, O
q = hidden_size * hidden_size
k = hidden_size * (num_kv_heads * head_dim)
v = hidden_size * (num_kv_heads * head_dim)
o = hidden_size * hidden_size
attn = q + k + v + o

# MLP: two projections
mlp = hidden_size * intermediate_size * 2

# Per layer
block = attn + mlp
total = block * num_layers

# Embeddings + LM head (tied embeddings assumed)
total += vocab_size * hidden_size

# In MB at fp16
weights_mb = (total * 2) / (1024 ** 2)
```

---

## Fit Classification

| Ratio (total / VRAM) | Label | `gpu_memory_util` | Meaning |
|---|---|---|---|
| `< 55%` | `comfortable` | `0.90` | Safe, headroom for batching |
| `55% – 75%` | `tight` | `0.85` | Usable but watch fragmentation |
| `75% – 90%` | `very tight` | `0.80` | Risky, small batches only |
| `> 90%` | `unlikely` | `0.75` | Filtered out by default |

The `--allow-fits` flag controls which classes make it into `config.yaml`:

```bash
# Default: only comfortable
python generate_gpu_config.py a40 --allow-fits "comfortable"

# Include tighter fits
python generate_gpu_config.py a40 --allow-fits "comfortable,tight"

# Aggressive
python generate_gpu_config.py a40 --allow-fits "comfortable,tight,very tight"
```

---

## The `--concurrent-users` Cap

When you set `--concurrent-users 50`, the config generator does **not** just divide VRAM by 50. It:

1. Reserves 20% for overhead
2. Computes KV budget per user: `(vram - weights - reserve) / users`
3. Converts that budget back into a max `seq_len` per user
4. Caps the candidate `seq_lens` to that value

If even the smallest `seq_len` (e.g., 4096) exceeds the per-user budget, the model is **skipped** with reason `"insufficient KV budget for N concurrent users"`.

```python
# Conservative factor for scheduling fluctuations
raw_cap = int((kv_budget_per_user_mb / kv_per_token_mb) * 0.75)
```

---

## Vision-Specific Sizing (`generate_gpu_vision_config.py`)

Vision models add an **image token reserve** (default 1024 tokens) because image embeddings consume context window slots:

```python
usable_input_tokens = seq_len - target_max_output_tokens - image_token_reserve
```

Additional differences from the text config generator:

| Feature | Text (`generate_gpu_config.py`) | Vision (`generate_gpu_vision_config.py`) |
|---|---|---|
| Config extraction | `config.json` top-level or `text_config` | Recursive search for `text_config`, `llm_config`, `decoder_config`, etc. |
| Weight estimation | Text backbone only | Text + vision tower + video tower (if present) |
| Token reserve | None | `image_token_reserve` subtracted from usable tokens |
| vLLM extra args | None | Emits `--limit-mm-per-prompt` and `--mm-processor-kwargs` |
| Trust remote code | Hardcoded set | Auto-detected from `auto_map` or repo name (e.g., `internvl`) |

### Vision vLLM Extra Args Example

```yaml
vllm_extra_args:
  - "--limit-mm-per-prompt"
  - '{"image": 1}'
  - "--mm-processor-kwargs"
  - '{"crop_size": 448}'
```

These are passed through to the vLLM CLI by `evaluate_vision.py`'s `VLlmManager`.

---

## GPU Target Support

```python
TARGET_GPU_VRAM_GB = {
    "a40": 48,
    "l40s": 48,
    "3090": 24,
    "a100": 80,
    "h200_sxm": 141,
    "b200": 180,
}
```

- `l40s` uses the same 48 GB class as `a40`
- `3090` is the most constrained at 24 GB
- `b200` (Blackwell) gets CUDA 12.8 wheels during setup
