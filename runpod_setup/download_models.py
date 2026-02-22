#!/usr/bin/env python3
"""
Download all configured models to /workspace/models/

Usage:
    export HF_TOKEN=your_token
    python download_models.py
"""
import os
import sys
import yaml
from pathlib import Path
from huggingface_hub import snapshot_download
from tqdm import tqdm


def load_config():
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_model(name: str, config: dict, hf_token: str):
    """Download a single model."""
    repo_id = config["repo"]
    local_path = Path(config["local_path"])
    
    print(f"\n📥 Downloading: {name}")
    print(f"   Repository: {repo_id}")
    print(f"   Local path: {local_path}")
    
    if local_path.exists() and any(local_path.iterdir()):
        print(f"   ⚠️  Directory exists, checking for updates...")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_path),
            local_dir_use_symlinks=False,
            resume_download=True,
            token=hf_token,
            exclude=["*.msgpack", "*.h5", "*.ot"]  # Skip unused formats
        )
        
        # Calculate size
        total_size = sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file())
        size_gb = total_size / (1024**3)
        
        print(f"   ✅ Complete ({size_gb:.1f} GB)")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def main():
    """Download all models."""
    config = load_config()
    
    # Get HF token
    hf_token = os.getenv("HF_TOKEN") or config.get("hf_token")
    if not hf_token:
        print("❌ Error: HF_TOKEN not set")
        print("   Set it as environment variable or in config.yaml")
        sys.exit(1)
    
    models = config.get("models", {})
    
    print("=" * 60)
    print("  Model Download")
    print("=" * 60)
    print(f"\nTotal models to download: {len(models)}")
    print("This will take 1-3 hours depending on your connection.")
    print("")
    
    # Download each model
    success_count = 0
    for name, model_config in models.items():
        if download_model(name, model_config, hf_token):
            success_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"  Complete: {success_count}/{len(models)} models downloaded")
    print("=" * 60)
    
    if success_count < len(models):
        sys.exit(1)


if __name__ == "__main__":
    main()
