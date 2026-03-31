#!/usr/bin/env python3
"""
Download 10-15 small models for Johnson's Rule demo
===================================================
Downloads a curated set of small, fast models suitable for demonstrating
batch scheduling with Johnson's Rule.
"""

import subprocess
import sys
from pathlib import Path

# Curated list: small models, diverse families, fast downloads
MODELS = [
    # Qwen family (already have some, but ensure all variants)
    'Qwen/Qwen2.5-0.5B-Instruct',
    'Qwen/Qwen2.5-1.5B-Instruct',
    'Qwen/Qwen2.5-3B-Instruct',
    'Qwen/Qwen2.5-7B-Instruct',
    'Qwen/Qwen3-0.6B',
    'Qwen/Qwen3-8B',

    # Llama 3.2 (small variants)
    'meta-llama/Llama-3.2-1B-Instruct',
    'meta-llama/Llama-3.2-3B-Instruct',

    # Phi family (Microsoft, very small and fast)
    'microsoft/phi-2',
    'microsoft/Phi-3-mini-4k-instruct',

    # TinyLlama (smallest)
    'TinyLlama/TinyLlama-1.1B-Chat-v1.0',

    # Gemma (Google, small variants)
    'google/gemma-2b-it',

    # StableLM
    'stabilityai/stablelm-2-zephyr-1_6b',
]

STORAGE_PATH = './models_batch'

def download_model(model_name: str):
    """Download a model using huggingface-cli."""
    print(f'\n{"="*70}')
    print(f'Downloading: {model_name}')
    print(f'{"="*70}')

    try:
        # Use huggingface-cli to download
        cmd = [
            'huggingface-cli', 'download',
            model_name,
            '--local-dir', f'{STORAGE_PATH}/{model_name}',
            '--local-dir-use-symlinks', 'False',
        ]

        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f'✓ {model_name} downloaded successfully')
        return True

    except subprocess.CalledProcessError as e:
        print(f'✗ Failed to download {model_name}: {e}')
        return False
    except FileNotFoundError:
        print('✗ huggingface-cli not found. Install with: pip install huggingface_hub[cli]')
        return False

def check_existing(model_name: str) -> bool:
    """Check if model already exists locally."""
    model_path = Path(STORAGE_PATH) / model_name
    if model_path.exists() and any(model_path.iterdir()):
        print(f'✓ {model_name} already exists, skipping')
        return True
    return False

def main():
    print('='*70)
    print('Model Download Script for Johnson\'s Rule Demo')
    print('='*70)
    print(f'Target: {len(MODELS)} models')
    print(f'Storage: {STORAGE_PATH}')
    print()

    Path(STORAGE_PATH).mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = 0

    for model in MODELS:
        if check_existing(model):
            skipped += 1
            continue

        if download_model(model):
            downloaded += 1
        else:
            failed += 1

    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    print(f'Downloaded: {downloaded}')
    print(f'Skipped (already exists): {skipped}')
    print(f'Failed: {failed}')
    print(f'Total available: {downloaded + skipped}')
    print('='*70)

    if downloaded + skipped >= 3:
        print('\n✓ Ready to run experiments')
    else:
        print(f'\n✗ Only {downloaded + skipped} models available, need at least 3')
        sys.exit(1)

if __name__ == '__main__':
    main()
