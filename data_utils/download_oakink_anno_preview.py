#!/usr/bin/env python3
"""Download anno_preview files from OakInk-v2 dataset."""

from huggingface_hub import snapshot_download
import os

print("Downloading anno_preview files from OakInk-v2 dataset...")
print(f"Target directory: {os.path.abspath('./OakInk-v2-anno_preview')}")

# Get HF token from environment variable
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("Warning: HF_TOKEN not found. You may hit rate limits.")
    print("Set it with: export HF_TOKEN='your_token_here'")

snapshot_download(
    repo_id="kelvin34501/OakInk-v2",
    repo_type="dataset",
    allow_patterns="anno_preview/*",
    local_dir="./OakInk-v2-anno_preview",
    local_dir_use_symlinks=False,
    token=hf_token
)

print("\nDownload complete!")
print(f"Files saved to: {os.path.abspath('./OakInk-v2-anno_preview/anno_preview')}")
