import copy
import tempfile
import unittest
from gpulab.store import Store
from gpulab.catalog import config
from gpulab.runner import DEMO_GPU,demo_result
from gpulab.util import fingerprint

class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.store=Store(self.tmp.name)
        self.rid=self.store.create(dict(name='Baseline',description='',tags=[],notes='',hardware={'gpu':'original'}),dict(gpu=DEMO_GPU,host={}),config(),['matrix'],DEMO_GPU,True)
    def tearDown(self):self.tmp.cleanup()
    def test_edits_preserve_evidence_and_results(self):
        result=demo_result('matrix',config());self.store.result(self.rid,result)
        self.store.edit_metadata(self.rid,dict(name='Corrected',hardware={'gpu':'corrected'}))
        run=self.store.get(self.rid)
        self.assertEqual(run['revision'],2)
        self.assertEqual(run['snapshot']['gpu'],DEMO_GPU)
        self.assertEqual(run['results'],[result])
        self.assertEqual(run['audit'][-1]['before']['hardware']['gpu'],'original')
    def test_measurements_cannot_be_edited_as_metadata(self):
        with self.assertRaises(ValueError):self.store.edit_metadata(self.rid,dict(results=[]))
    def test_bundle_round_trip(self):
        self.store.result(self.rid,demo_result('matrix',config()))
        self.store.update(self.rid,status='completed')
        bundle=self.store.export(self.rid);other=self.store.import_run(bundle)
        self.assertNotEqual(self.rid,other)
        copied=self.store.get(other)
        self.assertEqual(copied['results'],bundle['results'])
        self.assertEqual(copied['source'],'import')
    def test_tampered_bundle_rejected(self):
        bundle=self.store.export(self.rid);bundle['config']['repeats']=99
        with self.assertRaises(ValueError):self.store.import_run(bundle)
    def test_invalid_import_does_not_create_partial_run(self):
        bundle=self.store.export(self.rid);bundle['results']=[dict(test_id='not-a-test',status='passed')]
        bundle['bundle_fingerprint']=fingerprint({k:bundle[k] for k in ('snapshot','config','results')})
        before=len(self.store.list())
        with self.assertRaises(ValueError):self.store.import_run(bundle)
        self.assertEqual(len(self.store.list()),before)
    def test_stale_process_preserves_completed_results(self):
        self.store.result(self.rid,demo_result('matrix',config()))
        self.store.update(self.rid,status='running',pid=99999999,marker='oldboot:123')
        self.store.recover();run=self.store.get(self.rid)
        self.assertEqual(run['status'],'interrupted');self.assertEqual(len(run['results']),1)
    def test_path_traversal_rejected(self):
        with self.assertRaises(ValueError):self.store.directory('../outside')
    def test_measurement_version_survives_import(self):
        bundle=self.store.export(self.rid)
        bundle['snapshot']['app_version']='0.0.1'
        bundle['bundle_fingerprint']=fingerprint({k:bundle[k] for k in ('snapshot','config','results')})
        rid=self.store.import_run(bundle)
        self.assertEqual(self.store.get(rid)['app_version'],'0.0.1')

if __name__=='__main__':unittest.main()
