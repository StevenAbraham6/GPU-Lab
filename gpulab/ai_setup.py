"""Provision the public Qwen model in an existing vendor runtime."""
import subprocess
from pathlib import Path

MODEL = 'Qwen/Qwen2.5-7B-Instruct'
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'


def setup(root, vendor):
    python = Path(root) / 'backends' / vendor / 'bin' / 'python'
    if not python.is_file():
        raise ValueError(f'No {vendor} backend found at {python}. Install the GPU backend first.')
    model = Path(root) / 'models' / 'Qwen2.5-7B-Instruct'
    model.mkdir(parents=True, exist_ok=True)
    print(f'Installing AI dependencies in {vendor} backend...', flush=True)
    subprocess.run([str(python), '-m', 'pip', 'install', 'transformers>=4.43,<5', 'safetensors'], check=True)
    subprocess.run([str(python), '-c', 'import torch, transformers, safetensors'], check=True)
    print('Downloading Qwen2.5-7B-Instruct (approximately 15 GB); no account or token required.', flush=True)
    download = ("import sys; from huggingface_hub import snapshot_download; "
                "snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], "
                "token=False, allow_patterns=['*.json','*.safetensors','*.txt','LICENSE'])")
    subprocess.run([str(python), '-c', download, MODEL, REVISION, str(model)], check=True)
    required = ['config.json', 'tokenizer.json'] + [f'model-0000{i}-of-00004.safetensors' for i in range(1, 5)]
    missing = [name for name in required if not (model / name).is_file()]
    if missing:
        raise ValueError('Model download incomplete: ' + ', '.join(missing))
    print(f'AI setup complete. Model directory: {model}', flush=True)
