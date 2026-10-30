import hmac
import json
import os
import re
import secrets
import threading
import time
from contextlib import closing
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs,urlparse
from .catalog import TESTS,PROFILES,config,select_tests
from .compare import comparison
from .devices import discover,select_gpu
from .preflight import check
from .reports import report_csv,report_html,report_pdf
from .runner import DEMO_GPU,cancel,continuation,create_run
from .store import Store,validate_metadata
from .util import now
from .updates import MaintenanceConflict,Updates

class LabServer(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,address,store,token=None):
        super().__init__(address,Handler)
        self.store=store;self.token=token;self.session=secrets.token_urlsafe(32)
        self.inventory=None;self.inventory_time=0;self.inventory_lock=threading.Lock()
        self.instance=secrets.token_hex(16)
        self.updates=Updates(store,self.instance,f"http://127.0.0.1:{self.server_port}/api/health")
    def devices(self):
        with self.inventory_lock:
            if self.inventory is None or time.monotonic()-self.inventory_time>30:
                self.inventory=discover();self.inventory_time=time.monotonic()
            return self.inventory

class Handler(BaseHTTPRequestHandler):
    server_version="GPU-Lab/0.1"
    def log_message(self,format,*args):
        # Do not log credentials or full request paths (tokens are accepted only in cookies/headers).
        print(f"{now()} {self.command} {self.path.split('?')[0]} {args[1] if len(args)>1 else ''}",flush=True)
    def send(self,status,data,ctype="application/json",filename=None,cookie=None):
        if not isinstance(data,bytes):data=json.dumps(data,allow_nan=False).encode()
        self.send_response(status);self.send_header("Content-Type",ctype);self.send_header("Content-Length",str(len(data)))
        self.send_header("Cache-Control","no-store");self.send_header("X-Content-Type-Options","nosniff")
        self.send_header("Referrer-Policy","no-referrer")
        self.send_header("Content-Security-Policy","default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        if filename:self.send_header("Content-Disposition",f'attachment; filename="{filename}"')
        if cookie:self.send_header("Set-Cookie",cookie)
        self.end_headers();self.wfile.write(data)
    def authorized(self):
        if not self.server.token:return True
        auth=self.headers.get("Authorization","")
        if auth.startswith("Bearer ") and hmac.compare_digest(auth[7:],self.server.token):return True
        cookie=SimpleCookie()
        try:cookie.load(self.headers.get("Cookie",""))
        except Exception:return False
        return "gpu_lab_session" in cookie and hmac.compare_digest(cookie["gpu_lab_session"].value,self.server.session)
    def body(self):
        if self.headers.get_content_type()!="application/json":raise ValueError("Content-Type must be application/json")
        size=int(self.headers.get("Content-Length","0"))
        if not 0<size<=32*1024**2:raise ValueError("JSON payload must be 1 byte to 32 MiB")
        value=json.loads(self.rfile.read(size))
        if not isinstance(value,dict):raise ValueError('JSON body must be an object')
        return value
    def do_GET(self):self.dispatch("GET")
    def do_POST(self):self.dispatch("POST")
    def do_PATCH(self):self.dispatch("PATCH")
    def dispatch(self,method):
        try:
            url=urlparse(self.path);path=url.path;query=parse_qs(url.query)
            if self.server.server_address[0] in ('127.0.0.1','::1'):
                hostname=urlparse('http://'+self.headers.get('Host','')).hostname
                if hostname not in ('127.0.0.1','localhost','::1'):
                    return self.send(403,{'error':'Unexpected Host header for a loopback workspace'})
            if path in ("/","/app.js","/settings.js","/style.css","/favicon.svg") and method=="GET":
                name={"/":"index.html"}.get(path,path[1:]);file=Path(__file__).parent/"static"/name
                ctype={".html":"text/html; charset=utf-8",".js":"text/javascript; charset=utf-8",".css":"text/css; charset=utf-8",".svg":"image/svg+xml"}[file.suffix]
                return self.send(200,file.read_bytes(),ctype)
            if method!="GET":
                origin=self.headers.get("Origin")
                if origin and urlparse(origin).netloc!=self.headers.get("Host"):
                    return self.send(403,{"error":"Cross-origin mutations are disabled"})
            if path=="/api/session":
                if method=="GET":return self.send(200,dict(auth_required=bool(self.server.token),authenticated=self.authorized()))
                if method=="POST":
                    body=self.body()
                    if self.server.token and not hmac.compare_digest(str(body.get("token","")),self.server.token):return self.send(401,{"error":"Incorrect access token"})
                    return self.send(200,{"ok":True},cookie=f"gpu_lab_session={self.server.session}; HttpOnly; SameSite=Strict; Path=/")
            if path=="/api/health" and method=="GET":
                from . import __version__
                with closing(self.server.store.connect()) as db:db.execute("SELECT 1").fetchone()
                return self.send(200,dict(ok=True,version=__version__,instance=self.server.instance,commit=self.server.updates.config.get("installed_commit")))
            if not self.authorized():return self.send(401,{"error":"Authentication required"})
            store=self.server.store
            if path=="/api/settings" and method=="GET":return self.send(200,self.server.updates.status())
            if path in ("/api/settings/check","/api/settings/update","/api/settings/restart") and method=="POST":
                self.body()
                return self.send(202,self.server.updates.start(path.rsplit("/",1)[1]))
            if path=="/api/catalog" and method=="GET":return self.send(200,dict(tests=TESTS,profiles=PROFILES,defaults=config()))
            if path=="/api/devices" and method=="GET":return self.send(200,self.server.devices())
            if path=="/api/runs" and method=="GET":return self.send(200,store.list())
            if path=="/api/preflight" and method=="POST":
                body=self.body();cfg=config(body.get("profile","quick"),**body.get("config",{}));tests=select_tests(body.get("tests"))
                gpu=DEMO_GPU if body.get("simulated") else select_gpu(self.server.devices(),body.get("gpu"))
                report=check(gpu,tests,cfg,store.root)
                active=[r for r in store.list() if r["gpu"]["id"]==gpu["id"] and r["status"] in ("queued","preflight","running","cancelling")]
                if active:report["checks"].append(dict(name="GPU reservation",state="warning",detail="Another GPU Lab run is active; new runs will queue behind it",tests=[]))
                return self.send(200,report)
            if path=="/api/runs" and method=="POST":
                body=self.body();cfg=config(body.get("profile","quick"),**body.get("config",{}))
                validate_metadata(body.get('metadata',{}))
                with self.server.updates.lock:
                    self.server.updates.ensure_idle()
                    rid=create_run(store,body.get("gpu"),body.get("tests"),cfg,simulated=bool(body.get("simulated")),allow_partial=bool(body.get("allow_partial")))
                if body.get("metadata"):store.edit_metadata(rid,body["metadata"])
                return self.send(201,{"id":rid})
            if path=="/api/import" and method=="POST":return self.send(201,{"id":store.import_run(self.body())})
            if path=="/api/compare" and method=="GET":
                return self.send(200,comparison(store.get(query["a"][0],samples=False),store.get(query["b"][0],samples=False)))
            match=re.fullmatch(r"/api/runs/([0-9a-f-]{36})(?:/(.*))?",path)
            if match:
                rid,action=match.groups();store.recover()
                if method=="GET" and not action:return self.send(200,store.get(rid))
                if method=="PATCH" and not action:return self.send(200,store.edit_metadata(rid,self.body()))
                if method=="POST" and action in ("cancel","resume","reproduce"):
                    self.body()
                    if action=="cancel":cancel(store,rid);return self.send(200,{"ok":True})
                    with self.server.updates.lock:
                        self.server.updates.ensure_idle()
                        next_id=continuation(store,rid,unfinished_only=action=="resume")
                    return self.send(201,{"id":next_id})
                if method=="GET" and action in ("report.pdf","report.html","results.csv","bundle.json"):
                    run=store.get(rid);stem="gpu-lab-"+rid[:8]
                    if action=="report.pdf":data,ctype=report_pdf(run),"application/pdf"
                    elif action=="report.html":data,ctype=report_html(run),"text/html; charset=utf-8"
                    elif action=="results.csv":data,ctype=report_csv(run),"text/csv; charset=utf-8"
                    else:data,ctype=json.dumps(store.export(rid),indent=2,allow_nan=False).encode(),"application/json"
                    return self.send(200,data,ctype,stem+"."+action.split(".")[-1])
                if method=="GET" and action and action.startswith("log/"):
                    name=action[4:]
                    if name not in ["worker.log","preflight.json"]+[t["id"]+".log" for t in TESTS]:raise ValueError("Invalid log name")
                    file=store.directory(rid)/name
                    return self.send(200,file.read_bytes()[-512000:] if file.exists() else b"Log not available yet","text/plain; charset=utf-8")
            self.send(404,{"error":"Not found"})
        except KeyError as e:self.send(404,{"error":f"Not found: {e}"})
        except MaintenanceConflict as e:self.send(409,{"error":str(e)})
        except (ValueError,TypeError) as e:self.send(400,{"error":str(e)})
        except (BrokenPipeError,ConnectionResetError):pass
        except Exception as e:
            import traceback;traceback.print_exc()
            self.send(500,{"error":str(e)})

def serve(store,host="127.0.0.1",port=8080,token=None):
    if host not in ("127.0.0.1","localhost","::1") and not token:
        raise ValueError("A non-loopback interface requires GPU_LAB_TOKEN or --token-file. Use an SSH tunnel for private access.")
    store.recover();server=LabServer((host,port),store,token)
    print(f"GPU Lab: http://{host}:{server.server_port} · data: {store.root}",flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
