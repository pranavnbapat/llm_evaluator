#!/usr/bin/env python3
"""
Static (GPU-free) model viability check for vLLM.

Given a Hugging Face repo ID or URL, this script:
  - downloads config.json metadata (no weights)
  - reports architecture + key dimensions
  - flags quantization config hints
  - estimates KV cache memory at 4k/8k/16k

This does NOT prove the model will run, but provides a high-confidence signal.
"""
import argparse
from typing import Any, Dict, Optional, List

import requests

TARGET_GPU_VRAM_GB = {
    "a40": 48,
    "l40s": 48,
    "3090": 24,
    "a100": 80,
    "h200_sxm": 141,
    "b200": 180,
}


def normalize_target_gpu(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    v = value.strip().lower().replace("-", "_")
    aliases = {
        "a40": "a40",
        "l40": "l40s",
        "l40s": "l40s",
        "l40_s": "l40s",
        "l40_sxm": "l40s",
        "3090": "3090",
        "rtx_3090": "3090",
        "rtx3090": "3090",
        "geforce_rtx_3090": "3090",
        "a100": "a100",
        "a100_sxm": "a100",
        "a100sxm": "a100",
        "h200": "h200_sxm",
        "h200_sxm": "h200_sxm",
        "h200sxm": "h200_sxm",
        "b200": "b200",
    }
    return aliases.get(v)


def detect_target_from_name(name: str, fallback: str) -> str:
    n = name.lower()
    if "h200" in n:
        return "h200_sxm"
    if "b200" in n or "gb200" in n or "blackwell" in n:
        return "b200"
    if "l40s" in n or "l40" in n:
        return "l40s"
    if "3090" in n:
        return "3090"
    if "a40" in n:
        return "a40"
    if "a100" in n:
        return "a100"
    return fallback


def normalize_model_ref(value: str) -> str:
    if value.startswith("http://") or value.startswith("https://"):
        parts = value.split("huggingface.co/", 1)
        if len(parts) == 2:
            repo = parts[1].strip("/")
            if repo:
                return repo
    return value


def fetch_config(repo_id: str, hf_token: Optional[str] = None) -> Dict[str, Any]:
    url = f"https://huggingface.co/{repo_id}/resolve/main/config.json"
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    resp = requests.get(url, timeout=30, headers=headers)
    resp.raise_for_status()
    return resp.json()


def fetch_params(repo_id: str, hf_token: Optional[str] = None) -> Optional[Dict[str, Any]]:
    url = f"https://huggingface.co/{repo_id}/resolve/main/params.json"
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    resp = requests.get(url, timeout=30, headers=headers)
    if resp.status_code == 200:
        return resp.json()
    return None


def estimate_kv_cache_mb(
    hidden_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    dtype_bytes: int,
) -> float:
    # KV cache per layer per token: 2 * num_kv_heads * head_dim
    # Total tokens: seq_len
    # Total layers: num_layers
    # Total bytes: 2 * num_layers * seq_len * num_kv_heads * head_dim * dtype_bytes
    total_bytes = 2 * num_layers * seq_len * num_kv_heads * head_dim * dtype_bytes
    return total_bytes / (1024 ** 2)


def estimate_weights_mb(cfg: Dict[str, Any]) -> Optional[float]:
    """Estimate weight memory from transformer config if possible."""
    try:
        hidden_size = cfg.get("hidden_size")
        num_layers = cfg.get("num_hidden_layers")
        num_heads = cfg.get("num_attention_heads")
        num_kv_heads = cfg.get("num_key_value_heads", num_heads)
        intermediate_size = cfg.get("intermediate_size")
        vocab_size = cfg.get("vocab_size")
        if not all([hidden_size, num_layers, intermediate_size, vocab_size, num_heads, num_kv_heads]):
            return None
        head_dim = hidden_size // num_heads

        # Attention: Q, K, V, O
        q = hidden_size * hidden_size
        k = hidden_size * (num_kv_heads * head_dim)
        v = hidden_size * (num_kv_heads * head_dim)
        o = hidden_size * hidden_size
        attn = q + k + v + o

        # MLP: two projections
        mlp = hidden_size * intermediate_size * 2

        block = attn + mlp
        total = block * num_layers

        # Embeddings + LM head (tied embeddings assumed)
        total += vocab_size * hidden_size

        return (total * 2) / (1024 ** 2)
    except Exception:
        return None


def classify_fit(total_mb: float, vram_gb: int) -> str:
    """Heuristic classification by VRAM headroom."""
    vram_mb = vram_gb * 1024
    ratio = total_mb / vram_mb
    if ratio < 0.55:
        return "comfortable"
    if ratio < 0.75:
        return "tight"
    if ratio < 0.90:
        return "very tight"
    return "unlikely"


def choose_max_len(seq_lens, weights_mb, hidden_size, num_layers, num_kv_heads, head_dim, dtype_bytes, vram_gb):
    vram_mb = vram_gb * 1024
    candidates = []
    for seq_len in seq_lens:
        kv_mb = estimate_kv_cache_mb(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            seq_len=seq_len,
            dtype_bytes=dtype_bytes,
        )
        total_mb = weights_mb + kv_mb
        ratio = total_mb / vram_mb
        candidates.append((seq_len, total_mb, ratio))

    # Prefer the largest seq_len that stays under 75% of VRAM
    safe = [c for c in candidates if c[2] <= 0.75]
    if safe:
        return max(safe, key=lambda x: x[0])

    # Otherwise pick the smallest that fits under 90%
    tight = [c for c in candidates if c[2] <= 0.90]
    if tight:
        return min(tight, key=lambda x: x[2])

    # Fallback to the smallest seq_len
    return min(candidates, key=lambda x: x[0])


def choose_gpu_mem_util(ratio: float) -> float:
    if ratio <= 0.55:
        return 0.90
    if ratio <= 0.70:
        return 0.85
    if ratio <= 0.80:
        return 0.80
    return 0.75


def main() -> int:
    parser = argparse.ArgumentParser(description="Static vLLM model viability check")
    parser.add_argument("model", help="HF repo ID or HF URL")
    parser.add_argument(
        "target_gpu_pos",
        nargs="?",
        default=None,
        help="Optional target GPU (a40|l40s|3090|a100|h200_sxm|b200). "
             "Allows: model_static_check.py <model> a100 --llm-optimize",
    )
    parser.add_argument("--dtype-bytes", type=int, default=2, help="Bytes per element (fp16/bf16=2)")
    parser.add_argument("--seq-lens", default="4096,8192,16384", help="Comma-separated seq lengths to estimate")
    parser.add_argument("--emit-config", action="store_true", help="Print a YAML config stub")
    parser.add_argument(
        "--target-gpu",
        default=None,
        help="Tune config for target GPU (a40|l40|l40s|3090|a100|a100sxm|h200|h200_sxm|b200)",
    )
    parser.add_argument("--quant", default=None, help="Quantization mode for stub (e.g., compressed-tensors, awq)")
    parser.add_argument("--llm-optimize", action="store_true", help="Use LLM to suggest optimal settings")
    parser.add_argument("--llm-base-url", default=None, help="Override LLM base URL")
    args = parser.parse_args()
    target_gpu = normalize_target_gpu(args.target_gpu or args.target_gpu_pos)
    if args.target_gpu and args.target_gpu_pos and normalize_target_gpu(args.target_gpu) != normalize_target_gpu(args.target_gpu_pos):
        print("Error: --target-gpu and positional target GPU disagree. Use only one or make them match.")
        return 2
    if (args.target_gpu or args.target_gpu_pos) and target_gpu is None:
        print("Error: invalid target GPU. Use one of: a40, l40s, 3090, a100, h200_sxm, b200")
        return 2

    repo_id = normalize_model_ref(args.model)
    env = load_env()
    hf_token = env.get("HF_TOKEN")
    cfg = fetch_config(repo_id, hf_token=hf_token)
    params = fetch_params(repo_id, hf_token=hf_token)

    architectures = cfg.get("architectures", [])
    torch_dtype = cfg.get("torch_dtype") or cfg.get("dtype")
    rope_scaling = cfg.get("rope_scaling")
    quant_cfg = cfg.get("quantization_config") or cfg.get("quantization") or cfg.get("quantization_method")

    # Prefer text_config for multi-modal models (e.g., Voxtral)
    text_cfg = cfg.get("text_config") or {}
    audio_cfg = cfg.get("audio_config") or {}
    vision_cfg = cfg.get("vision_config") or {}
    video_cfg = cfg.get("video_config") or {}

    hidden_size = text_cfg.get("hidden_size", cfg.get("hidden_size"))
    num_layers = text_cfg.get("num_hidden_layers", cfg.get("num_hidden_layers"))
    num_heads = text_cfg.get("num_attention_heads", cfg.get("num_attention_heads"))
    num_kv_heads = text_cfg.get("num_key_value_heads", cfg.get("num_key_value_heads", num_heads))
    max_pos = text_cfg.get("max_position_embeddings", cfg.get("max_position_embeddings"))
    rope_scaling = text_cfg.get("rope_scaling", rope_scaling) or text_cfg.get("rope_parameters")

    head_dim = None
    if hidden_size and num_heads:
        head_dim = hidden_size // num_heads

    print("=" * 72)
    print(f"Model: {repo_id}")
    print("=" * 72)
    print(f"architectures: {architectures}")
    print(f"hidden_size:   {hidden_size}")
    print(f"layers:        {num_layers}")
    print(f"heads:         {num_heads}")
    print(f"kv_heads:      {num_kv_heads}")
    print(f"head_dim:      {head_dim}")
    print(f"max_pos:       {max_pos}")
    print(f"torch_dtype:   {torch_dtype}")
    print(f"rope_scaling:  {rope_scaling}")
    print(f"quantization:  {quant_cfg}")
    if text_cfg:
        print(f"text_config:  present (using for KV estimate)")
    if audio_cfg:
        print(f"audio_config: present (not used for KV estimate)")
    if vision_cfg:
        print(f"vision_config: present (estimating weights)")
    if video_cfg:
        print(f"video_config: present (estimating weights)")
    if params:
        print(f"params.json:  present (checking for conflicts)")

    if not all([hidden_size, num_layers, num_heads, num_kv_heads, head_dim]):
        print("\nKV cache estimate: unavailable (missing config fields)")
        return 0

    print("\nEstimated KV cache (MB) at fp16/bf16 (text_config):")
    seq_lens = [int(x.strip()) for x in args.seq_lens.split(",") if x.strip().isdigit()]
    for seq_len in seq_lens:
        kv_mb = estimate_kv_cache_mb(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            seq_len=seq_len,
            dtype_bytes=args.dtype_bytes,
        )
        print(f"  seq_len {seq_len}: {kv_mb:,.1f} MB")

    # Estimate weights for text + audio + vision/video (when transformer-like fields exist)
    text_weights_mb = estimate_weights_mb(text_cfg if text_cfg else cfg)
    audio_weights_mb = estimate_weights_mb(audio_cfg) if audio_cfg else None
    vision_weights_mb = estimate_weights_mb(vision_cfg) if vision_cfg else None
    video_weights_mb = estimate_weights_mb(video_cfg) if video_cfg else None

    weights_mb = None
    if text_weights_mb is not None:
        weights_mb = text_weights_mb
        for extra in [audio_weights_mb, vision_weights_mb, video_weights_mb]:
            if extra is not None:
                weights_mb += extra
    if text_weights_mb is not None:
        print("\nHeuristic weights estimate (fp16/bf16):")
        print(f"  text:  {text_weights_mb:,.1f} MB")
        if audio_cfg:
            if audio_weights_mb is not None:
                print(f"  audio: {audio_weights_mb:,.1f} MB")
            else:
                print("  audio: unknown (insufficient fields)")
        if vision_cfg:
            if vision_weights_mb is not None:
                print(f"  vision:{vision_weights_mb:,.1f} MB")
            else:
                print("  vision: unknown (insufficient fields)")
        if video_cfg:
            if video_weights_mb is not None:
                print(f"  video: {video_weights_mb:,.1f} MB")
            else:
                print("  video: unknown (insufficient fields)")

    if weights_mb is not None:
        print("\nHeuristic fit estimate (weights + KV cache):")
        for seq_len in seq_lens:
            kv_mb = estimate_kv_cache_mb(
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                seq_len=seq_len,
                dtype_bytes=args.dtype_bytes,
            )
            total_mb = weights_mb + kv_mb
            a40 = classify_fit(total_mb, 48)
            l40s = classify_fit(total_mb, 48)
            rtx_3090 = classify_fit(total_mb, 24)
            a100 = classify_fit(total_mb, 80)
            h200_sxm = classify_fit(total_mb, 141)
            b200 = classify_fit(total_mb, 180)
            print(
                f"  seq_len {seq_len}: total ~{total_mb:,.1f} MB | "
                f"A40: {a40} | L40S: {l40s} | 3090: {rtx_3090} | "
                f"A100: {a100} | H200-SXM: {h200_sxm} | B200: {b200}"
            )
    else:
        print("\nHeuristic fit estimate: unavailable (insufficient config fields)")

    if params:
        params_dim = params.get("dim")
        params_layers = params.get("n_layers")
        params_heads = params.get("n_heads")
        params_kv_heads = params.get("n_kv_heads")
        params_head_dim = params.get("head_dim")

        mismatches = []
        if params_dim and hidden_size and params_dim != hidden_size:
            mismatches.append(f"dim {params_dim} != hidden_size {hidden_size}")
        if params_layers and num_layers and params_layers != num_layers:
            mismatches.append(f"n_layers {params_layers} != num_hidden_layers {num_layers}")
        if params_heads and num_heads and params_heads != num_heads:
            mismatches.append(f"n_heads {params_heads} != num_attention_heads {num_heads}")
        if params_kv_heads and num_kv_heads and params_kv_heads != num_kv_heads:
            mismatches.append(f"n_kv_heads {params_kv_heads} != num_key_value_heads {num_kv_heads}")
        if params_head_dim and head_dim and params_head_dim != head_dim:
            mismatches.append(f"head_dim {params_head_dim} != derived head_dim {head_dim}")

        if mismatches:
            print("\nWARNING: params.json differs from config.json:")
            for m in mismatches:
                print(f"  - {m}")

    if args.emit_config:
        print("\nYAML config stub:")
        repo_id = normalize_model_ref(args.model)
        model_key = repo_id.split("/")[-1].lower().replace("-", "_").replace(".", "_")
        suffix = "fp16"
        quant = args.quant
        if quant:
            suffix = quant.replace("-", "_")

        # Choose max_model_len and gpu_memory_util deterministically
        max_len = 4096
        gpu_mem = 0.85
        if weights_mb is not None and all([hidden_size, num_layers, num_kv_heads, head_dim]):
            if target_gpu:
                vram_gb = TARGET_GPU_VRAM_GB.get(target_gpu, 80)
                seq_len, total_mb, ratio = choose_max_len(
                    seq_lens=seq_lens,
                    weights_mb=weights_mb,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    dtype_bytes=args.dtype_bytes,
                    vram_gb=vram_gb,
                )
                max_len = seq_len
                gpu_mem = choose_gpu_mem_util(ratio)
                if classify_fit(total_mb, vram_gb) == "unlikely":
                    label = target_gpu.upper().replace("_SXM", "-SXM")
                    print(
                        f"\nYAML config stub: no safe config emitted for {label} "
                        "(heuristic fit: unlikely)."
                    )
                    return 0

        print(f"{model_key}_{suffix}_{target_gpu or 'default'}:")
        print(f"  name: \"{repo_id}-{'FP16' if not quant else quant.upper()}\"")
        print(f"  repo: \"{repo_id}\"")
        print(f"  local_path: \"/workspace/models/{model_key}_{suffix}_{target_gpu or 'default'}\"")
        print(f"  quant: {('null' if not quant else '\"' + quant + '\"')}")
        print(f"  dtype: \"float16\"")
        print(f"  max_model_len: {max_len}")
        print(f"  gpu_memory_util: {gpu_mem}")

    if args.llm_optimize:
        prompt = build_llm_prompt(
            repo_id=repo_id,
            architectures=architectures,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_pos=max_pos,
            torch_dtype=torch_dtype,
            quant_cfg=quant_cfg,
            weights_mb=weights_mb,
            seq_lens=seq_lens,
            dtype_bytes=args.dtype_bytes,
            target_gpu=target_gpu,
        )
        response = call_llm(prompt, base_url=args.llm_base_url)
        normalized = normalize_llm_yaml(
            response,
            repo_id=repo_id,
            weights_mb=weights_mb,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype_bytes=args.dtype_bytes,
            target_gpu=target_gpu,
        )
        print("\nLLM recommendation (raw):")
        print(response)
        if normalized:
            print("\nLLM recommendation (normalized for config.yaml):")
            print(normalized)
        elif target_gpu:
            label = target_gpu.upper().replace("_SXM", "-SXM")
            print(
                "\nLLM recommendation (normalized for config.yaml): "
                f"no safe config emitted for {label} (heuristic fit: unlikely)."
            )

    return 0


def load_env(path: str = ".env") -> Dict[str, str]:
    env = {}
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip()
    except FileNotFoundError:
        pass
    return env


def build_llm_prompt(
    repo_id: str,
    architectures,
    hidden_size,
    num_layers,
    num_heads,
    num_kv_heads,
    head_dim,
    max_pos,
    torch_dtype,
    quant_cfg,
    weights_mb,
    seq_lens,
    dtype_bytes,
    target_gpu: Optional[str] = None,
) -> str:
    kv_estimates = {}
    totals_by_seq = {}
    if all([hidden_size, num_layers, num_kv_heads, head_dim]):
        for seq_len in seq_lens:
            kv_mb = estimate_kv_cache_mb(
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                seq_len=seq_len,
                dtype_bytes=dtype_bytes,
            )
            kv_estimates[seq_len] = kv_mb
            if weights_mb is not None:
                totals_by_seq[seq_len] = {
                    "total_mb": weights_mb + kv_mb,
                    "a40_ratio": (weights_mb + kv_mb) / (48 * 1024),
                    "l40s_ratio": (weights_mb + kv_mb) / (48 * 1024),
                    "3090_ratio": (weights_mb + kv_mb) / (24 * 1024),
                    "a100_ratio": (weights_mb + kv_mb) / (80 * 1024),
                    "h200_sxm_ratio": (weights_mb + kv_mb) / (141 * 1024),
                    "b200_ratio": (weights_mb + kv_mb) / (180 * 1024),
                }

    lines = [
        "You are an expert in vLLM deployment and GPU memory sizing.",
        "Given the following model metadata and memory estimates, recommend:",
        "1) max_model_len",
        "2) gpu_memory_util",
        "3) Whether the model should run comfortably, tight, or unlikely on A40 (48GB), "
        "L40S (48GB), RTX 3090 (24GB), A100 (80GB), H200-SXM (141GB), and B200 (180GB).",
        "",
        "Use these heuristics:",
        "- KV cache grows with seq_len, layers, hidden_size, kv_heads, head_dim.",
        "- Prefer the largest seq_len that stays under 70–75% of VRAM for stable runs.",
        "- If between 75–85%, mark as tight.",
        "- If >90%, mark as unlikely.",
        "",
        f"Model metadata:",
        f"- repo: {repo_id}",
        f"- architecture: {architectures}",
        f"- hidden_size: {hidden_size}",
        f"- num_hidden_layers: {num_layers}",
        f"- num_attention_heads: {num_heads}",
        f"- num_key_value_heads: {num_kv_heads}",
        f"- head_dim: {head_dim}",
        f"- max_position_embeddings: {max_pos}",
        f"- dtype: {torch_dtype}",
        f"- quantization: {quant_cfg}",
        "",
        f"Estimated weights (MB): {weights_mb}",
        "Estimated KV cache (MB):",
    ]
    for seq_len in seq_lens:
        kv = kv_estimates.get(seq_len)
        if kv is not None:
            lines.append(f"  {seq_len}: {kv:.1f}")
            totals = totals_by_seq.get(seq_len)
            if totals:
                lines.append(
                    f"    total_mb={totals['total_mb']:.1f}, "
                    f"a40_ratio={totals['a40_ratio']:.3f}, "
                    f"l40s_ratio={totals['l40s_ratio']:.3f}, "
                    f"3090_ratio={totals['3090_ratio']:.3f}, "
                    f"a100_ratio={totals['a100_ratio']:.3f}, "
                    f"h200_sxm_ratio={totals['h200_sxm_ratio']:.3f}, "
                    f"b200_ratio={totals['b200_ratio']:.3f}"
                )
        else:
            lines.append(f"  {seq_len}: unknown")
    lines.append("")
    if target_gpu:
        label = target_gpu.upper().replace("_SXM", "-SXM")
        lines.append(f"Output one YAML block for {label} only.")
    else:
        lines.append("Output four YAML blocks (A40, A100, H200-SXM, B200) for gpu_runtime/config.yaml.")
    lines.append("Formatting rules:")
    lines.append("- Use lower-case keys with two-space indentation.")
    lines.append("- Use quant: null for FP16 (not 'none').")
    lines.append("- Use dtype: \"float16\" unless the model requires BF16.")
    lines.append("- local_path must be under /workspace/models/ and unique per block.")
    lines.append("- Use max_model_len and gpu_memory_util consistent with your own memory estimates.")
    lines.append("- For each target GPU, compute usage_ratio = (weights + kv_at_max_model_len) / vram.")
    lines.append("- Choose gpu_memory_util from usage_ratio using this mapping:")
    lines.append("  usage_ratio <= 0.55 -> 0.90")
    lines.append("  usage_ratio <= 0.70 -> 0.85")
    lines.append("  usage_ratio <= 0.80 -> 0.80")
    lines.append("  otherwise -> 0.75")
    lines.append("- Include a multi-line notes field using YAML block scalar (|).")
    lines.append("")
    lines.append("Each block must include:")
    lines.append("- name")
    lines.append("- repo")
    lines.append("- local_path")
    lines.append("- quant")
    lines.append("- dtype")
    lines.append("- max_model_len")
    lines.append("- gpu_memory_util")
    lines.append("- notes (brief)")
    return "\n".join(lines)


def call_llm(prompt: str, base_url: Optional[str] = None) -> str:
    env = load_env()
    base = base_url or env.get("LLM_URL", "").rstrip("/")
    model = env.get("LLM", "")
    api_key = env.get("LLM_API_KEY", "")

    if not base or not model:
        return "LLM not configured (missing LLM_URL or LLM in .env)."

    url = f"{base}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 500,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"LLM call failed: {type(e).__name__}: {e}"


def extract_yaml_blocks(text: str) -> List[str]:
    blocks = []
    in_block = False
    current = []
    for line in text.splitlines():
        if line.strip().startswith("```yaml"):
            in_block = True
            current = []
            continue
        if in_block and line.strip().startswith("```"):
            blocks.append("\n".join(current).strip())
            in_block = False
            current = []
            continue
        if in_block:
            current.append(line)
    return [b for b in blocks if b]


def parse_simple_yaml_block(block: str) -> List[Dict[str, Any]]:
    """
    Parse one code fence that may contain multiple top-level YAML blocks.
    Splits when encountering a new "name:" after collecting fields.
    """
    blocks: List[Dict[str, Any]] = []
    data: Dict[str, Any] = {}
    lines = block.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.strip().startswith("#"):
            i += 1
            continue
        if ":" not in line:
            i += 1
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if key == "name" and data:
            blocks.append(data)
            data = {}

        if value == "|":
            i += 1
            notes_lines = []
            while i < len(lines) and (lines[i].startswith("  ") or lines[i].startswith("\t")):
                notes_lines.append(lines[i].lstrip())
                i += 1
            data[key] = "\n".join(notes_lines).rstrip()
            continue
        data[key] = value.strip('"').strip("'")
        i += 1

    if data:
        blocks.append(data)
    return blocks


def parse_int(value: Any, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def parse_float(value: Any, default: float) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def normalize_llm_yaml(
    text: str,
    repo_id: str,
    weights_mb: Optional[float] = None,
    hidden_size: Optional[int] = None,
    num_layers: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
    dtype_bytes: int = 2,
    target_gpu: Optional[str] = None,
) -> str:
    blocks = extract_yaml_blocks(text)
    if not blocks:
        return ""
    out_lines = []
    idx = 0
    for block in blocks:
        parsed_blocks = parse_simple_yaml_block(block)
        for data in parsed_blocks:
            idx += 1
            name = data.get("name", f"{repo_id}")
            target = detect_target_from_name(name, target_gpu or f"block{idx+1}")
            if target_gpu and target != target_gpu:
                continue
            model_key = repo_id.split("/")[-1].lower().replace("-", "_").replace(".", "_")
            suffix = "fp16"
            quant = data.get("quant", "null").lower()
            if quant and quant not in {"null", "none"}:
                suffix = quant.replace("-", "_")
            top_key = f"{model_key}_{suffix}_{target}"

            # Normalize fields
            repo = data.get("repo", repo_id)
            local_path = f"/workspace/models/{top_key}"
            dtype = "float16"
            max_model_len = parse_int(data.get("max_model_len", "4096"), 4096)
            gpu_memory_util = parse_float(data.get("gpu_memory_util", "0.8"), 0.8)
            notes = data.get("notes", "").strip()
            if notes and not notes.startswith("-"):
                notes = "\n".join([f"- {line}" for line in notes.splitlines()])

            if (
                target in TARGET_GPU_VRAM_GB
                and weights_mb is not None
                and all([hidden_size, num_layers, num_kv_heads, head_dim])
            ):
                kv_mb = estimate_kv_cache_mb(
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    seq_len=max_model_len,
                    dtype_bytes=dtype_bytes,
                )
                vram_gb = TARGET_GPU_VRAM_GB[target]
                fit_label = classify_fit(weights_mb + kv_mb, vram_gb)
                if target_gpu and fit_label == "unlikely":
                    # Explicit target requested but heuristic fit is unlikely:
                    # do not emit a runnable config block.
                    continue
                ratio = (weights_mb + kv_mb) / (vram_gb * 1024)
                gpu_memory_util = choose_gpu_mem_util(ratio)

            gpu_memory_util = min(max(gpu_memory_util, 0.5), 0.95)
            out_lines.append(f"{top_key}:")
            out_lines.append(f"  name: \"{name}\"")
            out_lines.append(f"  repo: \"{repo}\"")
            out_lines.append(f"  local_path: \"{local_path}\"")
            out_lines.append(f"  quant: {('null' if quant in {'null','none'} else '\"' + quant + '\"')}")
            out_lines.append(f"  dtype: \"{dtype}\"")
            out_lines.append(f"  max_model_len: {max_model_len}")
            out_lines.append(f"  gpu_memory_util: {gpu_memory_util:.2f}")
            if notes:
                out_lines.append("  notes: |")
                for line in notes.splitlines():
                    out_lines.append(f"    {line}")
            out_lines.append("")
    return "\n".join(out_lines).rstrip()


if __name__ == "__main__":
    raise SystemExit(main())
