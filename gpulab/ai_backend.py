"""Local, single-GPU Llama inference soak. No downloads or CPU offload."""
import hashlib
import time
from pathlib import Path

from .catalog import metric
from .probe import torch_device

PROMPTS = (
    "Explain how a heat pump works in clear language.",
    "Write a short guide to planning a productive team meeting.",
    "Describe the differences between a lake, a river, and an ocean.",
    "Summarize the benefits and limitations of public transportation.",
)
OUTPUT_TOKENS = 64


def model_fingerprint(path):
    """Hash contents, not paths, so copied identical models can be compared."""
    files = sorted(p for p in path.iterdir() if p.is_file() and
                   (p.suffix == '.safetensors' or p.name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja')))
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode() + b'\0')
        with file.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def summarize(requests):
    total_tokens = sum(r['tokens'] for r in requests)
    total_time = sum(r['seconds'] for r in requests)
    metrics = [metric('completed requests', len(requests), 'requests', 'context'),
               metric('output tokens per second', total_tokens / total_time, 'tokens/s', 'higher'),
               metric('time to first token', sum(r['first'] for r in requests) / len(requests), 's', 'lower')]
    if len(requests) >= 2:
        count = max(1, len(requests) // 4)
        for label, subset in (('early', requests[:count]), ('late', requests[-count:])):
            metrics.append(metric(label + ' output speed', sum(r['tokens'] for r in subset) / sum(r['seconds'] for r in subset), 'tokens/s', 'higher'))
    return metrics


def inference(gpu, cfg):
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    import torch

    path = Path(cfg['ai_model_path']).expanduser().resolve()
    config = AutoConfig.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    rope = getattr(config, 'rope_scaling', None) or {}
    if ((config.model_type, config.hidden_size, config.num_hidden_layers) != ('llama', 4096, 32)
            or rope.get('rope_type', rope.get('type')) != 'llama3'):
        raise ValueError('Expected local Llama 3.1 8B Instruct model files; model architecture does not match')
    fingerprint = model_fingerprint(path)
    torch, runtime, dev = torch_device(gpu)
    runtime.set_device(dev)
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch.float16,
                                                  local_files_only=True, trust_remote_code=False)
    model.to(dev).eval()
    prompts = []
    for prompt in PROMPTS:
        encoded = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True,
                                                 return_tensors='pt')
        prompts.append(encoded.to(dev))
    settings = dict(model='Meta Llama 3.1 8B Instruct', model_sha256=fingerprint, precision='fp16',
                    max_output_tokens=OUTPUT_TOKENS, decoding='greedy, fixed token count',
                    prompt_set='gpu-lab-ai-v1', input_tokens=[int(p.shape[-1]) for p in prompts],
                    soak_seconds=cfg['soak_seconds'], torch_version=str(torch.__version__),
                    transformers_version=transformers.__version__)

    def request(tokens):
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(input_ids=tokens, use_cache=True)
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            runtime.synchronize()
            first = time.perf_counter() - started
            cache = output.past_key_values
            for _ in range(1, OUTPUT_TOKENS):
                output = model(input_ids=next_token, past_key_values=cache, use_cache=True)
                next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                cache = output.past_key_values
            runtime.synchronize()
        return dict(tokens=OUTPUT_TOKENS, seconds=time.perf_counter()-started, first=first)

    request(prompts[0])  # Warm-up is intentionally outside the measured period.
    runtime.reset_peak_memory_stats(dev)
    started_wall = time.time()
    start = time.monotonic()
    requests = []
    failure = None
    while time.monotonic() - start < cfg['soak_seconds']:
        try:
            requests.append(request(prompts[len(requests) % len(prompts)]))
        except Exception as exc:
            failure = str(exc)[:2000]
            break
    end_wall = time.time()
    metrics = summarize(requests) if requests else [metric('completed requests', 0, 'requests', 'context')]
    metrics.extend([metric('failed requests', int(failure is not None), 'requests', 'lower'),
                    metric('peak PyTorch allocated VRAM', runtime.max_memory_allocated(dev)/1024**2, 'MiB', 'context')])
    return [dict(name='Llama 3.1 8B inference', status='error' if failure else 'passed', reason=failure,
                 settings=settings, metrics=metrics, intervals=[],
                 measurement_window=dict(start=started_wall, end=end_wall),
                 note='Fixed-length greedy generation; per-request timeouts are not enforced. The overall test timeout remains active.')]
