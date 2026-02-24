# Model Evaluation Troubleshooting Guide

A guide documenting technical issues encountered during LLM evaluation on RunPod with A40 GPUs and their solutions.

---

## Table of Contents

1. [vLLM Startup Issues](#1-vllm-startup-issues)
2. [GPU Memory Issues](#2-gpu-memory-issues)
3. [Model Loading Issues](#3-model-loading-issues)
4. [Evaluation Script Issues](#4-evaluation-script-issues)
5. [Best Practices Summary](#5-best-practices-summary)

---

## 1. vLLM Startup Issues

### Issue 1.1: vLLM Timeout Too Short

**Problem:** Large models (especially AWQ quantized) take 10-15 minutes to load due to disk I/O and kernel compilation, exceeding default timeout.

**Error:**
```
❌ Timeout waiting for vLLM
```

**Root Cause:** Default 900s (15 min) timeout insufficient for 23GB+ model files on NVMe storage.

**Solution:** Increased timeout to 1800s (30 minutes).

**Monitoring:**
```bash
tail -f /tmp/vllm_model_name.log
```

---

## 2. GPU Memory Issues

### Issue 2.1: CUDA Out of Memory

**Problem:** Large models (Mixtral 47B, Mistral 24B) exceed A40 48GB capacity in FP16 mode.

**Error:**
```
torch.OutOfMemoryError: CUDA out of memory
Tried to allocate 1.75 GiB. GPU has 44.42 GiB total, 497 MiB free
```

**Root Cause:** 
- Mixtral-8x7B: 47B params × 2 bytes = ~94 GB needed (FP16)
- Mistral-Small-24B: 24B params × 2 bytes = ~48 GB needed (FP16)
- A40 has 48GB but needs headroom for KV cache and activations

**Solutions:**

1. **Reduce memory settings:**
```yaml
gpu_memory_util: 0.60
max_model_len: 4096
```

2. **Use quantized models:**
```yaml
# Original (90GB, fails on A40):
repo: "mistralai/Mixtral-8x7B-Instruct-v0.1"
quant: null

# AWQ Quantized (23GB, fits on A40):
repo: "TheBloke/Mixtral-8x7B-Instruct-v0.1-AWQ"
quant: "awq"
```

3. **Clear CUDA cache between models:**
```python
import torch
torch.cuda.empty_cache()
```

4. **Set PyTorch memory allocator config:**
```python
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

---

### Issue 2.2: GPU Memory Fragmentation

**Problem:** After running multiple models, CUDA memory becomes fragmented causing allocation failures despite sufficient total memory.

**Symptom:** Model should fit (e.g., 23GB < 44GB available) but allocation fails.

**Root Cause:** PyTorch CUDA allocator doesn't release memory back to OS between models.

**Solution:** Combined approaches above + force cache clear between model evaluations.

---

## 3. Model Loading Issues

### Issue 3.1: Wrong Quantization Type

**Problem:** Model's actual quantization format doesn't match config specification.

**Error:**
```
Quantization method specified in model config (compressed-tensors) 
does not match the quantization method specified (awq)
```

**Root Cause:** Different quant formats exist (AWQ, GPTQ, compressed-tensors) and repos may use different formats than expected.

**Solution:** Use correct quant type for each model:
```yaml
quant: "awq"                    # Standard AWQ (TheBloke models)
quant: "compressed-tensors"     # Neural Magic format (some newer models)
quant: "gptq"                   # GPTQ format
quant: "awq_marlin"             # Optimized AWQ (faster inference)
```

---

### Issue 3.2: Model Loads But All Requests Fail

**Problem:** vLLM starts successfully but returns 0 successful responses.

**Context:** Mixtral-8x7B-Instruct-AWQ loaded successfully after ~15 min but failed to generate any valid responses across 120 questions × 3 runs (0/360 successful).

**Root Cause:** 
- AWQ format incompatible with specific vLLM version (0.15.1)
- Some AWQ models require `awq_marlin` format for proper inference
- Potential dtype mismatch (float16 vs bfloat16)

**Solutions:**
- Try `quant: "awq_marlin"` instead of `quant: "awq"` (vLLM suggests this in logs)
- Try GPTQ quantized version instead of AWQ
- Specify dtype explicitly: `dtype: "bfloat16"`

---

## 4. Evaluation Script Issues

### Issue 4.1: Database Duplicates on Re-run

**Problem:** Re-running evaluation creates duplicate entries for already-completed models.

**Root Cause:** SQLite INSERTs happen after each question; no uniqueness constraint on model_name + question_id + run_number.

**Check current data:**
```bash
sqlite3 evaluation_results.db \
  "SELECT model_name, COUNT(*) FROM evaluations GROUP BY model_name;"
```

**Solution:** Edit config.yaml to comment out completed models:
```yaml
# Comment out completed models:
# eurollm_9b:
#   ...

# Only run missing model:
mixtral_8x7b_awq:
  ...
```

---

## 5. Best Practices Summary

### Pre-Flight Checklist

```bash
# 1. Verify .env has tokens
cat runpod_setup/.env

# 2. Check model files are complete
ls -la /workspace/models/model_name/

# 3. Check GPU available
nvidia-smi

# 4. Clear any zombie processes
pkill -f "vllm serve"

# 5. Check disk space
df -h /workspace
```

### Model Selection Guidelines

| GPU VRAM | Max FP16 Model | Recommendation |
|----------|---------------|----------------|
| 48GB (A40) | ~40B params | Use AWQ/GPTQ for 30B+ models |
| 80GB (A100) | ~70B params | Can run most models FP16 |
| 24GB (RTX 3090) | ~20B params | Must quantize 13B+ models |

### Quantization Format Compatibility

| Format | vLLM Support | Typical Size | Notes |
|--------|-------------|--------------|-------|
| AWQ | ✅ Yes | 25% of original | Good accuracy/speed tradeoff |
| GPTQ | ✅ Yes | 25% of original | Widely supported |
| AWQ Marlin | ✅ Yes | 25% of original | Faster than AWQ |
| Compressed-Tensors | ✅ Yes | Varies | Neural Magic format |
| FP8 | ⚠️ Partial | 50% of original | H100 only |
| BNB 4-bit | ❌ No | 50% of original | Not supported by vLLM |

### Troubleshooting Commands

```bash
# Check vLLM logs for errors
tail -100 /tmp/vllm_model_name.log

# Monitor GPU in real-time
watch -n 1 nvidia-smi

# Check database contents
sqlite3 evaluation_results.db \
  "SELECT model_name, language, COUNT(*) FROM evaluations GROUP BY model_name, language;"

# Kill stuck vLLM
pkill -f "vllm serve"

# Test model loading manually
/workspace/llm_evaluator/.venv/bin/vllm serve /workspace/models/model_name \
  --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.75
```

---

## Summary of Key Fixes Applied

| Issue | Files Modified | Key Change |
|-------|---------------|------------|
| Type conversion | `evaluate.py`, `evaluate_context.py` | Added `int()`, `float()` for config values |
| vLLM path | `evaluate.py`, `evaluate_context.py` | Full path using `sys.executable` |
| Timeout | `evaluate.py`, `evaluate_context.py` | 900s → 1800s (30 min) |
| CUDA memory | `evaluate.py`, `evaluate_context.py` | `PYTORCH_CUDA_ALLOC_CONF`, `torch.cuda.empty_cache()` |
| Model size | `config.yaml` | Switched to AWQ quantized versions |
| Dtype handling | `evaluate.py`, `evaluate_context.py` | `model_config.get("dtype", "auto")` |

---

**Last Updated:** February 24, 2026
