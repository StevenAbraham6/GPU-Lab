import tempfile
import unittest
import os
import shutil
import subprocess
import sys
import shlex
from pathlib import Path
from unittest.mock import patch

from gpulab.ai_backend import model_fingerprint, summarize
from gpulab.catalog import config, select_tests
from gpulab.preflight import check
from gpulab.ai_setup import setup, select_backends, MODEL, REVISION


class AIWorkloadTests(unittest.TestCase):
    def test_profile_and_metrics(self):
        self.assertEqual([config(p)['ai_soak_seconds'] for p in ('quick', 'standard', 'extended')], [30, 300, 1800])
        self.assertIn('ai', select_tests(['all']))
        rows = {m['name']: m['value'] for m in summarize([
            dict(tokens=64, seconds=2, first=.2), dict(tokens=64, seconds=4, first=.4)])}
        self.assertAlmostEqual(rows['output tokens per second'], 128 / 6)
        self.assertAlmostEqual(rows['time to first token'], .3)
        self.assertEqual(rows['early output speed'], 32)
        self.assertEqual(rows['late output speed'], 16)

    def test_ai_workloads_are_independent_and_stability_profile_is_unchanged(self):
        self.assertEqual(select_tests(['ai']), ['ai'])
        self.assertEqual(select_tests(['ai_moe']), ['ai_moe'])
        self.assertIn('ai_moe', select_tests(['all']))
        self.assertEqual(config('standard')['soak_seconds'], 1800)
        self.assertEqual(config('extended', ai_soak_seconds=60)['ai_soak_seconds'], 60)

    def test_moe_preflight_is_independent_and_requires_all_shards(self):
        gpu = dict(id='gpu', name='Test GPU', vendor='nvidia')
        with tempfile.TemporaryDirectory() as root, patch('gpulab.preflight.sample', return_value={'power_w': None, 'temperature_c': None}), patch('gpulab.preflight.command', return_value=(0, '', '')):
            model = Path(root)/'models'/'OLMoE-1B-7B-0924-Instruct'
            model.mkdir(parents=True)
            for name in ['config.json', 'tokenizer.json'] + [f'model-{i:05d}-of-00003.safetensors' for i in range(1, 4)]:
                (model/name).touch()
            result = check(gpu, ['ai', 'ai_moe'], config(), root, execute=False)
            rows = {row['name']: row for row in result['checks']}
            self.assertEqual(rows['Qwen model files']['state'], 'blocked')
            self.assertEqual(rows['OLMoE model files']['state'], 'ready')
            (model/'model-00003-of-00003.safetensors').unlink()
            result = check(gpu, ['ai_moe'], config(), root, execute=False)
            self.assertEqual(next(row for row in result['checks'] if row['name'] == 'OLMoE model files')['state'], 'blocked')

    def test_setup_downloads_moe_only_when_selected(self):
        from gpulab.ai_setup import MOE_MODEL, MOE_REVISION
        with tempfile.TemporaryDirectory() as root:
            python = Path(root)/'backends/nvidia/bin/python'
            python.parent.mkdir(parents=True)
            python.touch()
            def run(args, check):
                if MOE_MODEL in args:
                    for name in ['config.json', 'tokenizer.json'] + [f'model-{i:05d}-of-00003.safetensors' for i in range(1, 4)]:
                        (Path(args[-1])/name).touch()
            with patch('gpulab.ai_setup.subprocess.run', side_effect=run) as calls:
                setup(root, 'nvidia', 'moe')
            download = calls.call_args_list[-1].args[0]
            self.assertIn(MOE_MODEL, download)
            self.assertIn(MOE_REVISION, download)

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

    def test_detects_unique_nvidia_and_amd_backends(self):
        inventory={'gpus':[{'vendor':'nvidia'},{'vendor':'amd'},{'vendor':'nvidia'},{'vendor':'intel'}]}
        self.assertEqual(select_backends('auto',inventory),['amd','nvidia'])
        self.assertEqual(select_backends('nvidia'),['nvidia'])
        self.assertEqual(select_backends('both'),['amd','nvidia'])
        with self.assertRaisesRegex(ValueError,'No NVIDIA or AMD GPU detected'):
            select_backends('auto',{'gpus':[]})

    def test_setup_with_both_vendors_downloads_only_once(self):
        with tempfile.TemporaryDirectory() as root:
            for vendor in ('nvidia','amd'):
                python=Path(root)/'backends'/vendor/'bin/python'
                python.parent.mkdir(parents=True)
                python.touch()
            def run(args, check):
                if MODEL in args:
                    model=Path(args[-1])
                    for name in ['config.json','tokenizer.json']+[f'model-0000{i}-of-00004.safetensors' for i in range(1,5)]:
                        (model/name).touch()
            with patch('gpulab.ai_setup.subprocess.run',side_effect=run) as calls:
                setup(root,['amd','nvidia'])
            self.assertEqual(sum(MODEL in call.args[0] for call in calls.call_args_list),1)
            self.assertEqual(sum('-m' in call.args[0] for call in calls.call_args_list),2)

    def test_new_installer_runs_ai_setup_for_both_backends(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            source=root/'source'
            source.mkdir()
            shutil.copyfile('install.sh',source/'install.sh')
            (source/'pyproject.toml').write_text('[project]\nname="test"\n')
            fakebin=root/'fakebin'
            fakebin.mkdir()
            uname=fakebin/'uname'
            uname.write_text('#!/bin/sh\necho Linux\n')
            uname.chmod(0o755)
            python=root/'python'
            python.write_text('''#!/usr/bin/env bash
set -eu
if [[ "$1" == '-I' ]]; then shift; fi
if [[ "$1" == '-m' && "$2" == 'venv' ]]; then mkdir -p "$3/bin"; cp "$0" "$3/bin/python"; exit 0; fi
if [[ "$1" == '-m' && "$2" == 'pip' ]]; then exit 0; fi
if [[ "$1" == '-m' && "$2" == 'gpulab' ]]; then printf '%s\\n' "$*" >> "$GPU_LAB_TEST_LOG"; exit 0; fi
exec '''+shlex.quote(sys.executable)+''' "$@"
''')
            python.chmod(0o755)
            log=root/'commands.log'
            env={**os.environ,'HOME':str(root),'PATH':str(fakebin)+os.pathsep+os.environ['PATH'],
                 'GPU_LAB_PYTHON':str(python),'GPU_LAB_PREFIX':str(root/'app'),
                 'GPU_LAB_HOME':str(root/'data'),'GPU_LAB_BIN':str(root/'bin'),'GPU_LAB_TEST_LOG':str(log)}
            result=subprocess.run(['bash',str(source/'install.sh'),'--backend','both','--no-system-packages','--skip-video-build'],env=env,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('setup-ai --backend both',log.read_text())


if __name__ == '__main__':
    unittest.main()
