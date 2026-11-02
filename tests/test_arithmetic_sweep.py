import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from gpulab.opencl_backend import OpenCL


class ArithmeticSweepTests(unittest.TestCase):
    def test_queued_work_counts_all_operations_and_checks_every_chain(self):
        backend = OpenCL.__new__(OpenCL)
        backend.np = np
        backend.queue = object()
        backend.buffer = lambda size: size
        events = []
        class Event:
            profile = SimpleNamespace(start=10, end=1010)
            def wait(self):
                self.waited = True
        def kernel(*args):
            event = Event()
            events.append(event)
            return event
        backend.k = {f'arithmetic_fp_{chains}': kernel for chains in (4, 8, 16, 32)}
        class CL:
            @staticmethod
            def enqueue_copy(queue, actual, out):
                ids = np.arange(64, dtype=np.uint32)
                base = np.stack([.01*(ids%17), .02*(ids%13), .03*(ids%11), .04*(ids%7)], axis=1).astype(np.float32)
                ref = np.stack([base + np.float32(v*.001) for v in range(actual.shape[1])], axis=1)
                for _ in range(1024):
                    ref = (ref.astype(np.float64)*float(np.float32(1.000001)) + float(np.float32(.000001))).astype(np.float32)
                actual[:] = ref
                return Event()
        backend.cl = CL
        for chains in (4, 8, 16, 32):
            with self.subTest(chains=chains):
                events.clear()
                fn, work, out, settings = backend.arithmetic_sweep_job(chains)
                self.assertEqual(len(events), 8)
                self.assertTrue(events[-1].waited)
                self.assertEqual(out, 262144 * (chains // 4) * 16)
                self.assertAlmostEqual(work, 262144 * 1024 * chains * 2 * 8 / 1e9)
                self.assertEqual(settings['independent_accumulators'], chains)
                self.assertEqual(settings['queue_depth'], 8)
                self.assertAlmostEqual(fn(), 8e-6)
                self.assertEqual(len(events), 16)

    def test_reports_batches_and_device_launches_separately(self):
        backend = OpenCL.__new__(OpenCL)
        backend.arithmetic_job = lambda kind: (None, 1, None, {})
        backend.arithmetic_sweep_job = lambda chains: (None, 8, None, {'queue_depth': 8})
        def measure(fn, cfg, work, unit, name, settings):
            return {'name': name, 'intervals': [{'dispatches': 3, 'wall_rate': 4}], 'metrics': []}
        with patch('gpulab.opencl_backend.timed_loop', side_effect=measure):
            cases = backend.arithmetic({})
        self.assertEqual([case['name'] for case in cases[:2]], ['FP32', 'INT32'])
        self.assertEqual(len(cases), 6)
        for case in cases[2:]:
            self.assertEqual(case['intervals'][0]['batches'], 3)
            self.assertEqual(case['intervals'][0]['dispatches'], 24)
            self.assertEqual(case['metrics'][0]['name'], 'elapsed throughput')
            self.assertEqual(case['metrics'][0]['value'], 4)
