# VRAM, Context Window, and Concurrency Guide

This note explains what consumes GPU memory, what context window means in practice, and how to estimate parallel-user capacity.

## 1) What Actually Sits in VRAM

When an LLM is running, VRAM is mainly used by:

1. Model weights  
- Fixed size once loaded.  
- Rough FP16 intuition: `7B ~ 14 GB`, `30B ~ 60 GB`.  
- Quantization reduces this.

2. KV cache (working memory)  
- Grows with active tokens.  
- Depends on sequence length, layers, hidden/KV dimensions, and precision.  
- Main reason long context causes OOM.

3. Runtime buffers  
- Non-zero overhead (scheduler, kernels, temporary tensors).

## 2) Context Window in Plain English

Context window = total tokens visible at once:

`system prompt + user input + retrieved RAG text + chat history + generated output`

So for an 8K model:

`input_tokens + output_tokens <= 8192`

Not input-only.

## 3) Token Intuition

- `1 token ~ 0.75 word` (very rough)  
- `100 tokens ~ short paragraph`  
- `1000 tokens ~ around a page` (roughly)

## 4) Why Context Affects Memory

KV cache scales with token count, so:

- 1K tokens -> small cache  
- 32K tokens -> large cache

Same model + same GPU can fail at longer context.

## 5) Single-Request RAG Example

Example token budget:

- System prompt: `120`
- User question: `20`
- RAG context: `3 x 500 = 1500`
- Formatting instructions: `80`

Input total: `1720`

If generation is `600` tokens, total active context is:

`1720 + 600 = 2320 tokens`

So:
- 4K context model -> OK
- 2K context model -> truncation/failure likely

## 6) Multi-User Concurrency: The Core Rule

Weights are shared across users, KV cache is not.

With `N` concurrent users:

`VRAM = model_weights + KV_user1 + KV_user2 + ... + KV_userN + runtime_overhead`

So concurrency is fundamentally a memory-capacity problem.

## 7) Practical Formula

Use this planning equation:

`concurrency ~= floor( available_vram_for_kv / kv_per_request )`

Where:

- `available_vram_for_kv = total_vram - model_weights - runtime_overhead`
- `kv_per_request` grows with prompt+output tokens

## 8) Best Model at 4K Context (Current Benchmark)

From the latest scoring results, the best overall model is:

- `devstral-small-2-24b-instruct-2512-b200`

To estimate concurrency at 4K, we must assume:

- FP16 weight footprint around `50-55 GB` for a 24B-class model (weights + serving overhead baseline).
- Runtime overhead reserve around `5 GB`.
- Effective KV per active 4K request around `1.2-2.0 GB` (depends on exact architecture and batching behavior).

These are planning estimates, not hard guarantees.

## 9) Estimated Parallel Users (4K Context)

| GPU | Total VRAM | Estimated free for KV | KV/request (4K) | Estimated concurrency |
| --- | ---: | ---: | ---: | ---: |
| B200 | 180 GB | ~120 GB | 1.2-2.0 GB | ~60-100 users |
| A100 SXM | 80 GB | ~20 GB | 1.2-2.0 GB | ~10-16 users |
| A40 | 48 GB | ~0 GB (for this FP16 24B model) | 1.2-2.0 GB | ~0 (does not fit reliably) |

### Important A40 Note

For this specific best model in FP16, A40 is generally not a safe fit for production concurrency at 4K.  
To run on A40, you typically need:

- quantized variant, and/or
- lower context, and/or
- lower concurrency targets.

## 10) Why You Can Still See Low VRAM in Some Workloads

If per-request batches are small or workload is sequential, VRAM may remain partially unused even when latency is high.  
That does not mean the GPU is idle; it means the bottleneck is often pipeline structure (batching/tokenization/request shape), not raw memory alone.

## 11) Throughput vs Depth Tradeoff

- High concurrency mode: short prompts + short outputs + many users
- Deep RAG mode: large retrieved context + long outputs + fewer users

The observed evaluation pattern is primarily deep-RAG style.

## 12) One-Sentence Summary for Stakeholders

Context window is not just a model property; at serving time it is a VRAM budget problem, and concurrency is capped by per-request KV memory.

## 13) A40-Friendly Alternatives (From Evaluated Models)

If the goal is to run reliably on A40 (48 GB) and still serve multiple parallel users at 4K context, these are practical choices from the evaluated models.

Notes:
- Names with `-b200` are experiment run labels; the model itself can still be deployed on other GPUs if memory permits.
- Numbers below are planning estimates with typical serving overhead and 4K context behavior.

| Model | Avg overall quality (scored results) | Typical weight footprint | Estimated free VRAM for KV on A40 | KV per 4K request | Estimated parallel users on A40 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `eurollm-9b-instruct-2512` | 0.8124 | ~18 GB (FP16) | ~25 GB | 0.6-1.2 GB | ~20-41 |
| `mistral-nemo-instruct-2407` | 0.8063 | ~24 GB (FP16) | ~19 GB | 0.8-1.4 GB | ~13-23 |
| `deepseek-r1-distill-qwen-7b` | 0.7749 | ~14 GB (FP16) | ~29 GB | 0.5-1.0 GB | ~29-58 |
| `deepseek-r1-distill-qwen-14b` | 0.7743 | ~28 GB (FP16) | ~15 GB | 0.8-1.5 GB | ~10-18 |

### Recommendation for A40

For the best quality-to-concurrency balance on A40 at 4K, `eurollm-9b-instruct-2512` is the strongest candidate in the current results.

### Reliability guardrails

To keep these concurrency ranges stable in production:
- keep retrieval chunks bounded (avoid very large context spikes),
- cap output length (`max_new_tokens`),
- use queueing/backpressure when concurrent long responses arrive together.
