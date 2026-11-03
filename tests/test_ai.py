import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gpulab.ai_backend import model_fingerprint, summarize
from gpulab.catalog import config, select_tests
from gpulab.preflight import check
from gpulab.ai_setup import setup, MODEL, REVISION


class AIWorkloadTests(unittest.TestCase):
    def test_profile_and_metrics(self):
        self.assertEqual([config(p)['soak_seconds'] for p in ('quick', 'standard', 'extended')], [30, 1800, 3600])
        self.assertIn('ai', select_tests(['all']))
        rows = {m['name']: m['value'] for m in summarize([
            dict(tokens=64, seconds=2, first=.2), dict(tokens=64, seconds=4, first=.4)])}
        self.assertAlmostEqual(rows['output tokens per second'], 128 / 6)
        self.assertAlmostEqual(rows['time to first token'], .3)
        self.assertEqual(rows['early output speed'], 32)
        self.assertEqual(rows['late output speed'], 16)

    def test_fingerprint_tracks_content_not_directory(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            for path in (Path(a), Path(b)):
                (path/'config.json').write_text('{}')
                (path/'model.safetensors').write_bytes(b'weights')
            self.assertEqual(model_fingerprint(Path(a)), model_fingerprint(Path(b)))
            (Path(b)/'model.safetensors').write_bytes(b'changed')
            self.assertNotEqual(model_fingerprint(Path(a)), model_fingerprint(Path(b)))

    def test_missing_model_blocks_only_ai(self):
        gpu = dict(id='gpu', name='Test GPU', vendor='nvidia')
        with tempfile.TemporaryDirectory() as root, patch('gpulab.preflight.sample', return_value={'power_w': None, 'temperature_c': None}), patch('gpulab.preflight.command', return_value=(0, '', '')):
            result = check(gpu, ['ai'], config(), root, execute=False)
        model_check = next(c for c in result['checks'] if c['name'] == 'Qwen model files')
        self.assertEqual(model_check['state'], 'blocked')
        self.assertEqual(model_check['tests'], ['ai'])

    def test_setup_downloads_public_pinned_model_into_default_directory(self):
        with tempfile.TemporaryDirectory() as root:
            python = Path(root)/'backends/nvidia/bin/python'
            python.parent.mkdir(parents=True)
            python.touch()
            def run(args, check):
                if MODEL in args:
                    model=Path(args[-1])
                    for name in ['config.json','tokenizer.json'] + [f'model-0000{i}-of-00004.safetensors' for i in range(1,5)]:
                        (model/name).touch()
            with patch('gpulab.ai_setup.subprocess.run', side_effect=run) as calls:
                setup(root,'nvidia')
            download=calls.call_args_list[-1].args[0]
            self.assertIn('token=False',download[2])
            self.assertIn(REVISION,download)
            self.assertEqual(download[-1],str(Path(root)/'models/Qwen2.5-7B-Instruct'))

    def test_setup_requires_existing_gpu_backend(self):
        with tempfile.TemporaryDirectory() as root, self.assertRaisesRegex(ValueError,'No nvidia backend'):
            setup(root,'nvidia')


if __name__ == '__main__':
    unittest.main()
