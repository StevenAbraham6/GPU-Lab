"""Local, single-GPU Qwen inference soak. No downloads or CPU offload."""
import hashlib
import time
from pathlib import Path

from .catalog import ai_cases, metric
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
    """Aggregate completed batches; elapsed is batch wall time, not summed per request."""
    total_tokens = sum(r['tokens'] for r in requests)
    total_time = sum(r['seconds'] for r in requests)
    count_requests = sum(r.get('batch', 1) for r in requests)
    metrics = [metric('completed requests', count_requests, 'requests', 'context'),
               metric('output tokens per second', total_tokens / total_time, 'tokens/s', 'higher'),
               metric('per-request output speed', sum(r['tokens'] / r.get('batch', 1) for r in requests) / total_time, 'tokens/s', 'higher'),
               metric('time to first token', sum(r['first'] * r.get('batch', 1) for r in requests) / count_requests, 's', 'lower'),
               metric('request completion time', sum(r['seconds'] * r.get('batch', 1) for r in requests) / count_requests, 's', 'lower')]
    if all('decode_tokens' in r for r in requests):
        decode_time = sum(r['seconds'] - r['first'] for r in requests)
        if decode_time > 0:
            metrics.append(metric('decode tokens per second', sum(r['decode_tokens'] for r in requests) / decode_time, 'tokens/s', 'higher'))
    if len(requests) >= 2:
        count = max(1, len(requests) // 4)
        for label, subset in (('early', requests[:count]), ('late', requests[-count:])):
            metrics.append(metric(label + ' output speed', sum(r['tokens'] for r in subset) / sum(r['seconds'] for r in subset), 'tokens/s', 'higher'))
    return metrics


def inference(gpu, cfg, model_kind="dense"):
    import gc
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    import torch

    moe = model_kind == 'moe'
    test = 'ai_moe' if moe else 'ai'
    name = 'OLMoE-1B-7B-0924-Instruct' if moe else 'Qwen2.5-7B-Instruct'
    path = Path(cfg['ai_moe_model_path'] if moe else cfg['ai_model_path']).expanduser().resolve()
    config = AutoConfig.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    expected = ('olmoe', 2048, 16) if moe else ('qwen2', 3584, 28)
    if (config.model_type, config.hidden_size, config.num_hidden_layers) != expected:
        raise ValueError(f'Expected local {name} model files; model architecture does not match')
    fingerprint = model_fingerprint(path)
    torch, runtime, dev = torch_device(gpu)
    runtime.set_device(dev)
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch.float16,
                                                  local_files_only=True, trust_remote_code=False)
    model.to(dev).eval()
    baseline = [tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True,
                                              return_tensors='pt') for prompt in PROMPTS]
    generated_prompts = {}
    def case_prompts(length):
        if length == 'baseline':
            return baseline
        if length not in generated_prompts:
            prompts = []
            for prompt in PROMPTS:
                # Identical input token length for every request; retain the end of the chat template.
                text = (prompt + ' ') * (length + 1)
                encoded = tokenizer.apply_chat_template([{'role': 'user', 'content': text}], add_generation_prompt=True, return_tensors='pt')
                if encoded.shape[-1] < length:
                    raise ValueError('Could not construct the requested prompt length')
                prompts.append(torch.cat((encoded[:, :1], encoded[:, -(length-1):]), dim=1))
            generated_prompts[length] = prompts
        return generated_prompts[length]

    def request(tokens, output_length):
        batch = int(tokens.shape[0])
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(input_ids=tokens, use_cache=True)
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            runtime.synchronize()
            first = time.perf_counter() - started
            cache = output.past_key_values
            for _ in range(1, output_length):
                output = model(input_ids=next_token, past_key_values=cache, use_cache=True)
                next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                cache = output.past_key_values
            runtime.synchronize()
        return dict(tokens=batch * output_length, decode_tokens=batch * (output_length - 1), batch=batch,
                    seconds=time.perf_counter()-started, first=first)

    cases = []
    for batch, input_length, output_length in ai_cases(cfg, test):
        legacy = batch == 1 and input_length == 'baseline' and output_length == 64
        title = ('OLMoE-1B-7B-0924-Instruct inference' if moe else 'Qwen2.5 7B inference') if legacy else f'Batch {batch} · input {input_length} · output {output_length}'
        settings = dict(model=name, model_sha256=fingerprint, precision='fp16', batch_size=batch,
                        input_length=input_length, max_output_tokens=output_length, decoding='greedy, fixed token count',
                        prompt_set='gpu-lab-ai-v1' if input_length == 'baseline' else 'gpu-lab-ai-length-v1',
                        soak_seconds=cfg.get('ai_soak_seconds', cfg['soak_seconds']), torch_version=str(torch.__version__),
                        transformers_version=transformers.__version__)
        start_wall = end_wall = None
        failure = None
        requests = []
        try:
            prompts = case_prompts(input_length)
            settings['input_tokens'] = [int(p.shape[-1]) for p in prompts]
            if max(settings['input_tokens']) + output_length > config.max_position_embeddings:
                raise ValueError('Input plus output exceeds model context limit')
            def run_batch(index):
                # Static batch of independent requests with equal-sized copies of one prompt.
                inputs = prompts[index % len(prompts)].to(dev).repeat(batch, 1)
                return request(inputs, output_length)
            run_batch(0)  # Each shape is warmed outside the timed period.
            runtime.reset_peak_memory_stats(dev)
            start_wall = time.time()
            start = time.monotonic()
            while time.monotonic() - start < settings['soak_seconds']:
                requests.append(run_batch(len(requests)))
            end_wall = time.time()
        except Exception as exc:
            failure = str(exc)[:2000]
            if start_wall is not None:
                end_wall = time.time()
        if failure:
            # Release exception tracebacks and cached blocks before starting another case.
            gc.collect()
            runtime.empty_cache()
        metrics = summarize(requests) if requests else [metric('completed requests', 0, 'requests', 'context')]
        metrics.append(metric('failed batches', int(failure is not None), 'batches', 'lower'))
        if start_wall is not None:
            metrics.append(metric('peak PyTorch allocated VRAM', runtime.max_memory_allocated(dev)/1024**2, 'MiB', 'context'))
        case = dict(name=title, status='error' if failure else 'passed', reason=failure,
                    settings=settings, metrics=metrics, intervals=[],
                    note='Static batch; copies of one prompt per batch. Fixed generation steps may continue after EOS. Times include prompt processing; per-request speed is output per request divided by batch completion time.')
        if start_wall is not None:
            case['measurement_window'] = dict(start=start_wall, end=end_wall)
        cases.append(case)
    return cases
