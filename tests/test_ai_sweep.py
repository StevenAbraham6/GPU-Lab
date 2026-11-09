import contextlib
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from gpulab.ai_backend import inference, summarize
from gpulab.catalog import ai_cases, config
from gpulab.runner import demo_result


class Tensor:
    def __init__(self, batch, length):
        self.shape = (batch, length)
    def __getitem__(self, key):
        if len(key) == 3: return self
        _, selection = key
        return Tensor(self.shape[0], len(range(*selection.indices(self.shape[1]))))
    def to(self, device): return self
    def repeat(self, batch, columns): return Tensor(batch, self.shape[1])
    def argmax(self, **kwargs): return Tensor(self.shape[0], 1)


class AIShapeTests(unittest.TestCase):
    def test_cartesian_sweep_and_validation(self):
        opts = {'batch_sizes': [1, 4], 'input_lengths': [128, 1024], 'output_lengths': [64, 256]}
        cfg = config(ai_moe_options=opts)
        self.assertEqual(len(ai_cases(cfg, 'ai_moe')), 8)
        self.assertEqual(ai_cases(cfg, 'ai'), [(1, 'baseline', 64)])
        for invalid in ([], [True], [3], [1, 1]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                config(ai_moe_options={**opts, 'batch_sizes': invalid})
        self.assertEqual(len(demo_result('ai_moe', cfg)['cases']), 8)

    def test_batch_metrics_use_batch_wall_time(self):
        result = {m['name']: m['value'] for m in summarize([
            dict(batch=4, tokens=256, decode_tokens=252, seconds=8, first=2),
            dict(batch=4, tokens=256, decode_tokens=252, seconds=8, first=2),
        ])}
        self.assertEqual(result['completed requests'], 8)
        self.assertEqual(result['output tokens per second'], 32)
        self.assertEqual(result['per-request output speed'], 8)
        self.assertEqual(result['decode tokens per second'], 42)
        self.assertEqual(result['request completion time'], 8)

    def test_inference_runs_cases_and_continues_after_out_of_memory(self):
        class Model:
            def __init__(self): self.calls=[]; self.fail=True
            def to(self, dev): return self
            def eval(self): return self
            def __call__(self, input_ids, **kwargs):
                self.calls.append(input_ids.shape)
                if input_ids.shape[0] == 4 and self.fail:
                    self.fail = False
                    raise RuntimeError('out of memory')
                return types.SimpleNamespace(logits=input_ids, past_key_values=object())
        model=Model()
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return Tensor(1, max(20, len(messages[0]['content'].split()) * 2))
        runtime=types.SimpleNamespace(set_device=lambda dev: None, synchronize=lambda: None,
            reset_peak_memory_stats=lambda dev: None, max_memory_allocated=lambda dev: 100,
            empty_cache=lambda: None)
        torch=types.SimpleNamespace(__version__='test', float16='fp16', inference_mode=contextlib.nullcontext,
            cat=lambda values, dim: Tensor(values[0].shape[0], sum(x.shape[1] for x in values)))
        trans=types.SimpleNamespace(__version__='test',
            AutoConfig=types.SimpleNamespace(from_pretrained=lambda *a, **k: types.SimpleNamespace(model_type='olmoe', hidden_size=2048, num_hidden_layers=16, max_position_embeddings=4096)),
            AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *a, **k: model),
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer()))
        with tempfile.TemporaryDirectory() as root:
            (Path(root)/'config.json').write_text('{}')
            opts={'batch_sizes':[4,1], 'input_lengths':[128], 'output_lengths':[64]}
            cfg=config(ai_moe_options=opts, ai_moe_model_path=root, ai_soak_seconds=1)
            with patch.dict('sys.modules', {'torch':torch, 'transformers':trans}), patch('gpulab.ai_backend.torch_device', return_value=(torch,runtime,'gpu')), patch('gpulab.ai_backend.time.monotonic', side_effect=[0, 0, 2]):
                cases=inference({},cfg,'moe')
        self.assertEqual([c['status'] for c in cases], ['error','passed'])
        self.assertIn('out of memory', cases[0]['reason'])
        self.assertEqual(cases[1]['settings']['input_tokens'], [128]*4)
        self.assertIn((4,128),model.calls)
        self.assertIn((1,128),model.calls)
        self.assertIn((1,1),model.calls)
        self.assertEqual(next(m['value'] for m in cases[1]['metrics'] if m['name']=='completed requests'),1)


if __name__ == '__main__': unittest.main()
