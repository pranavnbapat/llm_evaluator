#!/usr/bin/env python3
"""
Minimal vLLM model boot + inference test.

If the model loads and a short generation succeeds, the runtime is healthy.
"""
import argparse
import json
import sys
from typing import List, Optional

from vllm import LLM
from vllm.sampling_params import SamplingParams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM model boot test")
    parser.add_argument("model", help="HF repo ID, HF URL, or local path")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tokenizer-mode", default=None)
    parser.add_argument("--config-format", default=None)
    parser.add_argument("--load-format", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--prompt", default="Say 'hello' in one short sentence.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    return parser.parse_args()


def normalize_model_ref(value: str) -> str:
    """Accept HF URLs and return repo ID."""
    if value.startswith("http://") or value.startswith("https://"):
        parts = value.split("huggingface.co/", 1)
        if len(parts) == 2:
            repo = parts[1].strip("/")
            if repo:
                return repo
    return value


def build_llm(args: argparse.Namespace) -> LLM:
    model_ref = normalize_model_ref(args.model)
    kwargs = {
        "model": model_ref,
        "max_model_len": args.max_model_len,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.tokenizer_mode:
        kwargs["tokenizer_mode"] = args.tokenizer_mode
    if args.config_format:
        kwargs["config_format"] = args.config_format
    if args.load_format:
        kwargs["load_format"] = args.load_format
    if args.quantization:
        kwargs["quantization"] = args.quantization
    return LLM(**kwargs)


def main() -> int:
    args = parse_args()

    try:
        llm = build_llm(args)
    except Exception as e:
        if args.json:
            print(json.dumps({"status": "boot_failed", "error": str(e)}))
        else:
            print(f"BOOT FAILED: {e}")
        return 2

    params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    try:
        outputs = llm.generate([args.prompt], params)
        text = outputs[0].outputs[0].text if outputs else ""
    except Exception as e:
        if args.json:
            print(json.dumps({"status": "inference_failed", "error": str(e)}))
        else:
            print(f"INFERENCE FAILED: {e}")
        return 3

    if args.json:
        print(json.dumps({"status": "ok", "output": text}))
    else:
        print("OK")
        print(text.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
