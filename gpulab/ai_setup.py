"""Provision the public Qwen model in an existing vendor runtime."""
import subprocess
from pathlib import Path

MODEL = 'Qwen/Qwen2.5-7B-Instruct'
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
MOE_MODEL = 'allenai/OLMoE-1B-7B-0924-Instruct'
MOE_REVISION = '7f1c97f440f06ce36705e4f2b843edb5925f4498'


def select_backends(requested, inventory=None):
    if requested == 'both':
        return ['amd', 'nvidia']
    if requested != 'auto':
        return [requested]
    vendors = sorted({g.get('vendor') for g in (inventory or {}).get('gpus', [])} & {'nvidia', 'amd'})
    if not vendors:
        raise ValueError('No NVIDIA or AMD GPU detected. Check gpu-lab devices or choose --backend nvidia or --backend amd.')
    return vendors


def setup(root, vendors, model_choice="dense"):
    if isinstance(vendors, str):
        vendors = [vendors]
    runtimes = [(vendor, Path(root) / 'backends' / vendor / 'bin' / 'python') for vendor in vendors]
    for vendor, python in runtimes:
        if not python.is_file():
            raise ValueError(f'No {vendor} backend found at {python}. Install the GPU backend first.')
    model_name, repo, revision, shards = (('OLMoE-1B-7B-0924-Instruct', MOE_MODEL, MOE_REVISION, 3) if model_choice == 'moe' else ('Qwen2.5-7B-Instruct', MODEL, REVISION, 4))
    model = Path(root) / 'models' / model_name
    model.mkdir(parents=True, exist_ok=True)
    for vendor, python in runtimes:
        print(f'Installing AI dependencies in {vendor} backend...', flush=True)
        subprocess.run([str(python), '-m', 'pip', 'install', 'transformers>=4.46,<5', 'safetensors'], check=True)
        subprocess.run([str(python), '-c', 'import torch, transformers, safetensors'], check=True)
    print(f'Downloading {model_name} (approximately 14–15 GB); no account or token required.', flush=True)
    download = ("import sys; from huggingface_hub import snapshot_download; "
                "snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], "
                "token=False, allow_patterns=['*.json','*.safetensors','*.txt','LICENSE'])")
    subprocess.run([str(runtimes[0][1]), '-c', download, repo, revision, str(model)], check=True)
    required = ['config.json', 'tokenizer.json'] + [f'model-{i:05d}-of-{shards:05d}.safetensors' for i in range(1, shards + 1)]
    missing = [name for name in required if not (model / name).is_file()]
    if missing:
        raise ValueError('Model download incomplete: ' + ', '.join(missing))
    print(f'AI setup complete. Model directory: {model}', flush=True)
