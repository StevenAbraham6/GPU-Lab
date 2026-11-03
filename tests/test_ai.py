import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gpulab.ai_backend import model_fingerprint, summarize
from gpulab.catalog import config, select_tests
from gpulab.preflight import check


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
        model_check = next(c for c in result['checks'] if c['name'] == 'Llama model files')
        self.assertEqual(model_check['state'], 'blocked')
        self.assertEqual(model_check['tests'], ['ai'])


if __name__ == '__main__':
    unittest.main()
