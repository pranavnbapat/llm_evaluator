#!/usr/bin/env python3
"""
Batch-generate `models:` entries in runpod_setup/config.yaml for one target GPU.

Rules:
- Read HF repos from a plain text file (one repo per line).
- Estimate fit heuristically using model_static_check helpers.
- Include only models whose fit label is in --allow-fits (default: comfortable).
- Rewrite only the `models:` block in the config file.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_static_check import (  # noqa: E402
    TARGET_GPU_VRAM_GB,
    choose_gpu_mem_util,
    classify_fit,
    estimate_kv_cache_mb,
    estimate_weights_mb,
    fetch_config,
    load_env,
    normalize_target_gpu,
)


def parse_seq_lens(raw: str) -> List[int]:
    seq = sorted({int(x.strip()) for x in raw.split(",") if x.strip().isdigit()})
    if not seq:
        raise ValueError("No valid --seq-lens values provided.")
    return seq


def load_repo_list(path: Path) -> List[str]:
    repos: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        val = line.strip()
        if not val or val.startswith("#"):
            continue
        repos.append(val)
    return repos


def sanitize_repo_key(repo_id: str) -> str:
    return repo_id.split("/")[-1].lower().replace("-", "_").replace(".", "_")


def build_name(repo_id: str, gpu: str) -> str:
    base = repo_id.split("/")[-1].lower().replace("_", "-").replace(".", "-")
    return f"{base}-{gpu.replace('_', '-')}"


def extract_model_dims(cfg: Dict) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]:
    text_cfg = cfg.get("text_config") or {}
    hidden_size = text_cfg.get("hidden_size", cfg.get("hidden_size"))
    num_layers = text_cfg.get("num_hidden_layers", cfg.get("num_hidden_layers"))
    num_heads = text_cfg.get("num_attention_heads", cfg.get("num_attention_heads"))
    num_kv_heads = text_cfg.get("num_key_value_heads", cfg.get("num_key_value_heads", num_heads))
    head_dim = (hidden_size // num_heads) if (hidden_size and num_heads) else None
    return hidden_size, num_layers, num_heads, num_kv_heads, head_dim


def choose_len_by_fit(
    weights_mb: float,
    hidden_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype_bytes: int,
    seq_lens: List[int],
    target_vram_gb: int,
    allow_fits: List[str],
) -> Optional[Tuple[int, float, float, str]]:
    candidates: List[Tuple[int, float, float, str]] = []
    vram_mb = target_vram_gb * 1024
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
        fit = classify_fit(total_mb, target_vram_gb)
        ratio = total_mb / vram_mb
        candidates.append((seq_len, total_mb, ratio, fit))

    allowed = [c for c in candidates if c[3] in allow_fits]
    if not allowed:
        return None
    return max(allowed, key=lambda x: x[0])


def render_models_yaml(models: List[Dict[str, str]]) -> str:
    lines = ["models:"]
    for m in models:
        lines.append(f"  {m['key']}:")
        lines.append(f"    name: \"{m['name']}\"")
        lines.append(f"    repo: \"{m['repo']}\"")
        lines.append(f"    local_path: \"{m['local_path']}\"")
        lines.append("    quant: null")
        lines.append("    dtype: \"float16\"")
        lines.append(f"    max_model_len: {m['max_model_len']}")
        lines.append(f"    gpu_memory_util: {m['gpu_memory_util']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def replace_models_block(config_text: str, new_models_block: str) -> str:
    lines = config_text.splitlines(keepends=True)
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^models:\s*$", line.strip()):
            start = i
            break
    if start is None:
        raise ValueError("Could not find `models:` block in config file.")

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*:\s*$", lines[j]) and not lines[j].startswith("  "):
            end = j
            break

    before = "".join(lines[:start])
    after = "".join(lines[end:])
    return f"{before}{new_models_block}{after}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate GPU-specific models config block.")
    parser.add_argument("gpu", help="Target GPU: a40|a100|a100sxm|h200|h200_sxm|b200")
    parser.add_argument(
        "--repos-file",
        default="runpod_setup/model_repos.txt",
        help="Text file with one HF repo per line.",
    )
    parser.add_argument(
        "--config-file",
        default="runpod_setup/config.yaml",
        help="Config file where `models:` block will be replaced.",
    )
    parser.add_argument("--seq-lens", default="4096,8192,16384", help="Comma-separated candidate sequence lengths.")
    parser.add_argument("--dtype-bytes", type=int, default=2, help="fp16/bf16=2")
    parser.add_argument(
        "--allow-fits",
        default="comfortable",
        help="Comma-separated fit labels to include. Default: comfortable",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow writing an empty `models:` block.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print generated models block without writing file.")
    args = parser.parse_args()

    target_gpu = normalize_target_gpu(args.gpu)
    if not target_gpu or target_gpu not in TARGET_GPU_VRAM_GB:
        print("Error: invalid GPU. Use one of: a40, a100, h200_sxm, b200")
        return 2

    seq_lens = parse_seq_lens(args.seq_lens)
    allow_fits = [x.strip().lower() for x in args.allow_fits.split(",") if x.strip()]
    valid_fits = {"comfortable", "tight", "very tight", "unlikely"}
    if any(f not in valid_fits for f in allow_fits):
        print("Error: --allow-fits accepts: comfortable,tight,very tight,unlikely")
        return 2

    repos_file = Path(args.repos_file)
    config_file = Path(args.config_file)
    if not repos_file.exists():
        print(f"Error: repos file not found: {repos_file}")
        return 2
    if not config_file.exists():
        print(f"Error: config file not found: {config_file}")
        return 2

    env = load_env()
    hf_token = env.get("HF_TOKEN")
    target_vram_gb = TARGET_GPU_VRAM_GB[target_gpu]

    repos = load_repo_list(repos_file)
    if not repos:
        print(f"No repos found in {repos_file}")
        return 1

    generated: List[Dict[str, str]] = []
    skipped: List[Tuple[str, str]] = []

    for repo in repos:
        try:
            cfg = fetch_config(repo, hf_token=hf_token)
        except Exception as exc:
            skipped.append((repo, f"fetch failed: {type(exc).__name__}"))
            continue

        hidden_size, num_layers, _, num_kv_heads, head_dim = extract_model_dims(cfg)
        text_cfg = cfg.get("text_config") or {}
        weights_mb = estimate_weights_mb(text_cfg if text_cfg else cfg)

        if not all([hidden_size, num_layers, num_kv_heads, head_dim]) or weights_mb is None:
            skipped.append((repo, "missing dimensions or weight estimate"))
            continue

        chosen = choose_len_by_fit(
            weights_mb=weights_mb,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype_bytes=args.dtype_bytes,
            seq_lens=seq_lens,
            target_vram_gb=target_vram_gb,
            allow_fits=allow_fits,
        )
        if chosen is None:
            skipped.append((repo, f"no seq_len in allowed fits: {','.join(allow_fits)}"))
            continue

        max_model_len, _, ratio, fit = chosen
        model_key_base = sanitize_repo_key(repo)
        top_key = f"{model_key_base}_fp16_{target_gpu}"
        generated.append(
            {
                "key": top_key,
                "name": build_name(repo, target_gpu),
                "repo": repo,
                "local_path": f"/workspace/models/{top_key}",
                "max_model_len": str(max_model_len),
                "gpu_memory_util": f"{choose_gpu_mem_util(ratio):.2f}",
                "fit": fit,
            }
        )

    generated.sort(key=lambda x: x["key"])
    new_models_block = render_models_yaml(generated)

    if args.dry_run:
        print(new_models_block)
    else:
        if not generated and not args.allow_empty:
            print(
                "Refusing to overwrite config with empty models block. "
                "Use --allow-empty to force."
            )
            return 1
        original = config_file.read_text(encoding="utf-8")
        updated = replace_models_block(original, new_models_block)
        config_file.write_text(updated, encoding="utf-8")
        print(f"Updated models block in: {config_file}")

    print(
        f"\nTarget GPU: {target_gpu} ({target_vram_gb}GB) | "
        f"Included: {len(generated)} | Skipped: {len(skipped)}"
    )
    if generated:
        print("Included models:")
        for m in generated:
            print(f"  - {m['repo']} | max_model_len={m['max_model_len']} | fit={m['fit']}")
    if skipped:
        print("Skipped models:")
        for repo, reason in skipped:
            print(f"  - {repo} | {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
