**Core Rule**
vLLM loads a model if `config.json` → `architectures` maps to a supported backend.  
You do not check “is my exact model listed.” You check “does my model use a supported architecture.”

If the architecture is supported, the model usually runs.  
If not, it may fall back to Transformers (slower) or fail.

**The Only Test That Never Lies**
If the model boots and generates a short response, it works.  
Example:
```python
from vllm import LLM
llm = LLM(model="your/model")
```

---

**1) Architecture: The Real Gate**
Check `config.json`:
```
"architectures": [...]
```
Supported architectures → native kernels → fast.  
Unknown/custom architectures → Transformers fallback → 2–10× slower or fail.

Examples that are usually safe:
- `LlamaForCausalLM`
- `MistralForCausalLM`
- `Qwen2ForCausalLM`

---

**2) Context Length: KV Cache Is the VRAM Killer**
Check:
```
"max_position_embeddings"
```
KV cache grows with:
```
context × layers × hidden_size × kv_heads × head_dim
```
Two “7B” models can have wildly different VRAM needs if one has 4k context and the other has 128k.

---

**3) Hidden Size: The True Width**
Check:
```
"hidden_size"
```
Hidden size controls:
- KV cache size
- Activation memory
- Throughput

Example:
- 7B slim: hidden_size = 4096
- 7B wide: hidden_size = 8192  
Same “7B,” completely different runtime behavior.

---

**4) Number of Layers: Latency Multiplier**
Check:
```
"num_hidden_layers"
```
More layers = slower tokens/sec and more KV cache.

---

**5) GQA / MQA: Efficiency Switch**
Check:
```
"num_attention_heads"
"num_key_value_heads"
```
If `num_key_value_heads < num_attention_heads`, you have GQA/MQA.  
That means smaller KV cache and better long‑context efficiency.

---

**6) Rope Scaling: “Real” Long Context vs Marketing**
Check:
```
"rope_scaling"
```
This determines whether long context is native or scaled (YaRN, NTK, linear).  
Long contexts without correct rope scaling degrade quality.

---

**7) Quantization Must Match Model Config**
If `config.json` declares compressed‑tensors, vLLM will reject `--quantization awq`.  
Mismatch = immediate failure.

Check for:
```
"quantization_config"
```
and ensure your `quant` flag matches.

---

**8) Dtype**
Check:
```
"torch_dtype"
```
A40/A100 support BF16 and FP16.  
If BF16 is required and your GPU doesn’t support it, you pay conversion overhead or fail.

---

**9) Vision/Audio Models: Extra Tokens, Extra VRAM**
Vision models often include:
```
"vision_config": {"image_size": ..., "patch_size": ...}
```
Visual tokens scale roughly as:
```
tokens ≈ (image_size / patch_size)^2
```
Examples:
- 224 → 256 tokens
- 448 → 1024 tokens
- 896 → 4096 tokens

Audio models may include long sequence encoders that also add KV and activation memory.

---

**10) Tokenizer / Load Format Flags (Mistral‑style)**
Some official Mistral repos require:
```
--tokenizer_mode mistral
--config_format mistral
--load_format mistral
```
If these are missing, vLLM may fail to load or will use an incompatible tokenizer.

---

**11) Chat Templates: Hidden Source of Failures**
If `/v1/chat/completions` fails, try `/v1/completions`.  
Some models do not ship with a usable chat template.

---

**Example: Why Two “7B” Models Behave Differently**
Model A:
- hidden_size: 4096
- layers: 32
- context: 32k
- GQA: yes

Model B:
- hidden_size: 8192
- layers: 40
- context: 128k
- no GQA

Both are marketed as 7B.  
Runtime reality: Model B can behave like a 20B class model.

---

**A40 vs A100: The Real Constraints**
Key differences:
- VRAM: 48 GB vs 80 GB
- Memory bandwidth: ~696 GB/s vs ~2039 GB/s

Rule of thumb:
- VRAM = survival
- Bandwidth = speed
- Kernel support = stability

Compatibility is usually fine on both.  
The real question is: how large, how fast, how long context, and how many concurrent requests.
