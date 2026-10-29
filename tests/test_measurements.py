import math
import unittest
from gpulab.telemetry import integrate
from gpulab.catalog import config, select_tests
from gpulab.compare import comparison
from gpulab.video import cuda_index, encode_args, ffmpeg_env
from unittest.mock import patch

class EnergyTests(unittest.TestCase):
    def test_linear_power_with_interpolated_boundaries(self):
        samples=[dict(ts=i,power_w=100+10*i,power_scope="board") for i in range(5)]
        result=integrate(samples,.5,3.5)
        self.assertAlmostEqual(result['energy_j'],360)
        self.assertAlmostEqual(result['average_power_w'],120)

    def test_missing_endpoint_is_not_extrapolated(self):
        result=integrate([dict(ts=1,power_w=100),dict(ts=2,power_w=100)],0,3)
        self.assertIsNone(result['energy_j'])

    def test_sampling_gap_is_not_hidden(self):
        result=integrate([dict(ts=0,power_w=100),dict(ts=10,power_w=100)],0,10)
        self.assertEqual(result['coverage'],'sampling gap')

    def test_sensor_domain_change_invalidates_energy(self):
        result=integrate([dict(ts=0,power_w=100,power_scope="chip"),dict(ts=1,power_w=100,power_scope="board")],0,1)
        self.assertIsNone(result['energy_j'])

class ConfigurationTests(unittest.TestCase):
    def test_invalid_and_unbounded_inputs(self):
        for overrides in [dict(duration=-1),dict(buffer_mib=0),dict(repeats=100),dict(matrix_sizes=[100000]),dict(precisions=['fp4']),dict(max_temp=500),dict(duration=float('nan'))]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):config(**overrides)

    def test_removed_algorithms_are_not_part_of_suite(self):
        self.assertEqual(len(select_tests(['all'])),9)
        self.assertNotIn('algorithms',select_tests(['all']))
        self.assertEqual(select_tests(['compute']),['arithmetic','matrix'])

class ComparisonTests(unittest.TestCase):
    def run_data(self, value=10,direction='higher',scope='board',simulated=False,settings=None):
        return dict(id='id',app_version='0.1.0',snapshot={},simulated=simulated,results=[dict(test_id='matrix',cases=[dict(name='fixed shape',status='passed',settings=settings or {'precision':'fp32'},power={'power_scope':scope},metrics=[dict(name='throughput',value=value,unit='TFLOP/s',direction=direction)])])])

    def test_correct_direction_for_latency(self):
        c=comparison(self.run_data(10,'lower'),self.run_data(8,'lower'))
        self.assertAlmostEqual(c['rows'][0]['improvement_pct'],20)

    def test_different_workloads_are_not_ranked(self):
        c=comparison(self.run_data(),self.run_data(settings={'precision':'fp16'}))
        self.assertIsNone(c['rows'][0]['improvement_pct'])

    def test_simulated_results_never_rank_against_hardware(self):
        c=comparison(self.run_data(),self.run_data(simulated=True))
        self.assertIsNone(c['rows'][0]['improvement_pct'])

    def test_missing_data_is_visible(self):
        b=self.run_data();b['results']=[]
        c=comparison(self.run_data(),b)
        self.assertEqual(c['rows'][0]['reasons'],['Matching result missing'])

class DeviceSelectionTests(unittest.TestCase):
    def test_ffmpeg_receives_cuda_ordinal_not_pci_address(self):
        gpu=dict(vendor='nvidia',bus='0000:60:00.0',uuid='GPU-test',nvidia_index=2)
        with patch.dict('os.environ',{},clear=True):
            args=encode_args(gpu,'h264','quality',8000000)
            self.assertEqual(args[args.index('-gpu')+1],'0')
            self.assertEqual(ffmpeg_env(gpu)['CUDA_VISIBLE_DEVICES'],'GPU-test')
        with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'GPU-test'},clear=True):
            self.assertEqual(cuda_index(gpu),'0')
        with patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'0'},clear=True):
            with self.assertRaises(NotImplementedError):cuda_index(gpu)

if __name__=='__main__':unittest.main()
