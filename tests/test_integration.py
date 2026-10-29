import http.client
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import fcntl
from pathlib import Path
from gpulab.server import LabServer
from gpulab.store import Store
from gpulab.runner import create_run, continuation, cancel, DEMO_GPU
from gpulab.util import fingerprint
from gpulab.catalog import config
from gpulab.reports import report_pdf, report_csv, report_html

class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.store=Store(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def complete(self,rid):
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            run=self.store.get(rid)
            if run['status'] not in ('queued','preflight','running','cancelling'):return run
            time.sleep(.1)
        self.fail('Worker did not finish')
    def test_detached_run_reports_and_reproduction(self):
        rid=create_run(self.store,names=['arithmetic','matrix'],simulated=True)
        # No browser/server owns this worker; it finishes with only durable storage.
        run=self.complete(rid)
        self.assertEqual(run['status'],'completed')
        self.assertEqual(len(run['results']),2)
        self.assertTrue(report_pdf(run).startswith(b'%PDF'))
        self.assertIn(b'SIMULATED',report_html(run))
        self.assertIn(b'GFLOP/s',report_csv(run))
        second=continuation(self.store,rid,unfinished_only=False)
        self.assertEqual(self.complete(second)['parent'],rid)
    def test_cancel_while_queued_preserves_an_empty_run(self):
        lock=(self.store.root/'locks'/(fingerprint(DEMO_GPU['id'])+'.lock')).open('a+')
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            rid=create_run(self.store,names=['all'],simulated=True)
            cancel(self.store,rid)
            result=self.complete(rid)
            self.assertEqual(result['status'],'cancelled')
            self.assertEqual(result['results'],[])
        finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    def test_run_survives_http_server_shutdown(self):
        server=LabServer(('127.0.0.1',0),self.store)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        conn=http.client.HTTPConnection('127.0.0.1',server.server_port,timeout=10)
        conn.request('POST','/api/runs',json.dumps(dict(simulated=True,tests=['all'])),{'Content-Type':'application/json'})
        response=conn.getresponse();rid=json.loads(response.read())['id'];conn.close()
        server.shutdown();server.server_close();thread.join()
        self.assertEqual(self.complete(rid)['status'],'completed')
    def test_http_auth_origin_and_pdf_download(self):
        server=LabServer(('127.0.0.1',0),self.store,token='test-secret')
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        port=server.server_port
        def request(method,path,body=None,headers=None):
            conn=http.client.HTTPConnection('127.0.0.1',port,timeout=30)
            hs={'Content-Type':'application/json',**(headers or {})}
            conn.request(method,path,json.dumps(body) if body is not None else None,hs)
            response=conn.getresponse();value=response.read();code=response.status;hdr=dict(response.getheaders());conn.close();return code,value,hdr
        try:
            self.assertEqual(request('GET','/api/runs')[0],401)
            self.assertEqual(request('GET','/api/session',headers={'Host':'attacker.example'})[0],403)
            code,body,headers=request('POST','/api/session',dict(token='test-secret'))
            self.assertEqual(code,200);cookie=headers['Set-Cookie'].split(';')[0]
            hs={'Cookie':cookie}
            self.assertEqual(request('POST','/api/runs',dict(simulated=True),{**hs,'Origin':'https://evil.test'})[0],403)
            code,body,_=request('POST','/api/runs',dict(simulated=True,tests=['latency']),hs)
            self.assertEqual(code,201);rid=json.loads(body)['id'];self.complete(rid)
            code,pdf,headers=request('GET',f'/api/runs/{rid}/report.pdf',headers=hs)
            self.assertEqual(code,200);self.assertTrue(pdf.startswith(b'%PDF'));self.assertIn('attachment',headers['Content-Disposition'])
            code,_,_=request('GET',f'/api/runs/{rid}/log/../../lab.sqlite3',headers=hs)
            self.assertEqual(code,400)
        finally:server.shutdown();server.server_close();thread.join()

if __name__=='__main__':unittest.main()
