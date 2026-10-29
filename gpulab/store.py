"""SQLite is the authoritative journal. Raw records are append-only."""
import json
import sqlite3
import uuid
from pathlib import Path
from . import SCHEMA_VERSION, __version__
from .util import atomic_json, fingerprint, home, now, process_marker

ACTIVE = {"queued", "preflight", "running", "cancelling"}

def validate_metadata(changes):
    if not isinstance(changes,dict) or changes.keys()-{'name','description','tags','notes','hardware'}:
        raise ValueError('Only descriptive metadata is editable')
    for k in ('name','description','notes'):
        if k in changes and (not isinstance(changes[k],str) or len(changes[k])>20000):raise ValueError(f'Invalid {k}')
    if 'tags' in changes and (not isinstance(changes['tags'],list) or len(changes['tags'])>50 or any(not isinstance(t,str) or len(t)>100 for t in changes['tags'])):
        raise ValueError('Tags must be a list of short strings')
    if 'hardware' in changes and not isinstance(changes['hardware'],dict):raise ValueError('Hardware metadata must be an object')

class Store:
    def __init__(self, root=None):
        self.root = Path(root or home())
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "runs").mkdir(exist_ok=True)
        (self.root / "locks").mkdir(exist_ok=True)
        with self.connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
              id TEXT PRIMARY KEY, created TEXT NOT NULL, updated TEXT NOT NULL,
              status TEXT NOT NULL, metadata TEXT NOT NULL, snapshot TEXT NOT NULL,
              config TEXT NOT NULL, tests TEXT NOT NULL, gpu TEXT NOT NULL,
              simulated INTEGER NOT NULL DEFAULT 0, parent TEXT, revision INTEGER NOT NULL DEFAULT 1,
              cancel INTEGER NOT NULL DEFAULT 0, pid INTEGER, marker TEXT, heartbeat TEXT,
              current_test TEXT, error TEXT, source TEXT NOT NULL DEFAULT 'local',
              started_at TEXT, current_test_started_at TEXT);
            CREATE TABLE IF NOT EXISTS results (
              seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
              test_id TEXT NOT NULL, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS audit (
              seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
              at TEXT NOT NULL, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS telemetry (
              seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
              test_id TEXT, ts REAL NOT NULL, data TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS telemetry_run ON telemetry(run_id, ts);
            CREATE INDEX IF NOT EXISTS results_run ON results(run_id);
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(runs)")}
            if "started_at" not in columns:
                db.execute("ALTER TABLE runs ADD COLUMN started_at TEXT")
            if "current_test_started_at" not in columns:
                db.execute("ALTER TABLE runs ADD COLUMN current_test_started_at TEXT")

    def connect(self):
        db = sqlite3.connect(self.root / "lab.sqlite3", timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA busy_timeout=30000")
        return db

    def directory(self, run_id):
        if not run_id or any(c not in "0123456789abcdef-" for c in run_id) or len(run_id) != 36:
            raise ValueError("Invalid run ID")
        return self.root / "runs" / run_id

    def create(self, metadata, snapshot, cfg, tests, gpu, simulated=False, parent=None):
        validate_metadata(metadata)
        rid, stamp = str(uuid.uuid4()), now()
        self.directory(rid).mkdir()
        with self.connect() as db:
            db.execute("INSERT INTO runs(id,created,updated,status,metadata,snapshot,config,tests,gpu,simulated,parent) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                       (rid, stamp, stamp, "queued", json.dumps(metadata), json.dumps(snapshot),
                        json.dumps(cfg), json.dumps(tests), json.dumps(gpu), int(simulated), parent))
        self.audit(rid, dict(action="created", metadata=metadata))
        return rid

    def audit(self, rid, data):
        with self.connect() as db:
            db.execute("INSERT INTO audit(run_id,at,data) VALUES(?,?,?)", (rid, now(), json.dumps(data)))

    def get(self, rid, samples=True):
        self.directory(rid)
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
            if row is None:
                raise KeyError(rid)
            value = dict(row)
            for key in ("metadata", "snapshot", "config", "tests", "gpu"):
                value[key] = json.loads(value[key])
            value["simulated"] = bool(value["simulated"])
            value["results"] = [json.loads(r[0]) for r in db.execute("SELECT data FROM results WHERE run_id=? ORDER BY seq", (rid,))]
            value["audit"] = [dict(at=r[0], **json.loads(r[1])) for r in db.execute("SELECT at,data FROM audit WHERE run_id=? ORDER BY seq", (rid,))]
            value["telemetry"] = [dict(ts=r[0], test_id=r[1], **json.loads(r[2])) for r in db.execute("SELECT ts,test_id,data FROM telemetry WHERE run_id=? ORDER BY seq", (rid,))] if samples else []
        value.update(schema_version=SCHEMA_VERSION, app_version=value['snapshot'].get('app_version',__version__), report_app_version=__version__)
        return value

    def list(self):
        self.recover()
        with self.connect() as db:
            rows = db.execute("SELECT id,created,updated,status,metadata,gpu,simulated,current_test,revision,source FROM runs ORDER BY created DESC").fetchall()
        return [{**dict(r), "metadata": json.loads(r["metadata"]), "gpu": json.loads(r["gpu"]), "simulated": bool(r["simulated"])} for r in rows]

    def update(self, rid, **values):
        allowed = {"status", "pid", "marker", "heartbeat", "current_test", "error", "cancel", "started_at", "current_test_started_at"}
        if values.keys() - allowed:
            raise ValueError("Invalid update")
        values["updated"] = now()
        with self.connect() as db:
            db.execute("UPDATE runs SET " + ",".join(k + "=?" for k in values) + " WHERE id=?", (*values.values(), rid))

    def edit_metadata(self, rid, changes):
        validate_metadata(changes)
        # Read and merge inside one write transaction to avoid lost updates.
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT metadata FROM runs WHERE id=?", (rid,)).fetchone()
            if row is None:
                raise KeyError(rid)
            old = json.loads(row[0]); new = {**old, **changes}
            db.execute("UPDATE runs SET metadata=?,revision=revision+1,updated=? WHERE id=?", (json.dumps(new), now(), rid))
            db.execute("INSERT INTO audit(run_id,at,data) VALUES(?,?,?)", (rid, now(), json.dumps(dict(action="metadata edited", before=old, after=new))))
        return self.get(rid, samples=False)

    def result(self, rid, result):
        with self.connect() as db:
            db.execute("INSERT INTO results(run_id,test_id,data) VALUES(?,?,?)", (rid, result["test_id"], json.dumps(result, allow_nan=False)))
        atomic_json(self.directory(rid) / "results.json", self.get(rid, samples=False))

    def sample(self, rid, test_id, stamp, data):
        with self.connect() as db:
            db.execute("INSERT INTO telemetry(run_id,test_id,ts,data) VALUES(?,?,?,?)", (rid, test_id, stamp, json.dumps(data, allow_nan=False)))

    def samples(self, rid, start, end):
        with self.connect() as db:
            return [dict(ts=r[0], **json.loads(r[1])) for r in db.execute("SELECT ts,data FROM telemetry WHERE run_id=? AND ts BETWEEN ? AND ? ORDER BY ts", (rid, start, end))]

    def recover(self):
        with self.connect() as db:
            rows = db.execute("SELECT id,pid,marker,created FROM runs WHERE source='local' AND status IN ('queued','running','preflight','cancelling')").fetchall()
        from datetime import datetime, timezone
        for row in rows:
            # Newly created runs get a grace period for worker startup.
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(row["created"])).total_seconds()
            if row["pid"] and process_marker(row["pid"]) != row["marker"]:
                self.update(row["id"], status="interrupted", error="Worker stopped or host restarted. Completed test records are preserved.")
            elif not row["pid"] and age > 60:
                self.update(row["id"], status="interrupted", error="Worker did not start.")

    def export(self, rid):
        value = self.get(rid)
        value["bundle_fingerprint"] = fingerprint({"snapshot": value["snapshot"], "config": value["config"], "results": value["results"]})
        return value

    def import_run(self, value):
        from .catalog import BY_ID,config
        import math
        if value.get("schema_version") != SCHEMA_VERSION or not isinstance(value.get("results"), list):
            raise ValueError("Unsupported bundle schema")
        if len(value['results'])>len(BY_ID) or any(not isinstance(value.get(k),dict) for k in ('metadata','gpu','snapshot','config')):
            raise ValueError('Invalid bundle structure')
        if not isinstance(value['metadata'].get('name'),str) or not isinstance(value['gpu'].get('name'),str):
            raise ValueError('Run and GPU names are required')
        validate_metadata(value['metadata'])
        if not isinstance(value.get('tests'),list) or set(value['tests'])-set(BY_ID):raise ValueError('Unknown test in bundle')
        allowed_states={'passed','partial','unsupported','missing_dependency','failed_validation','error','cancelled'}
        seen=set()
        for result in value['results']:
            if not isinstance(result,dict) or result.get('test_id') not in BY_ID or result.get('test_id') in seen or result.get('status') not in allowed_states or not isinstance(result.get('cases'),list):
                raise ValueError('Invalid result identity/status')
            seen.add(result['test_id'])
            for case in result.get('cases',[]):
                if not isinstance(case,dict) or not isinstance(case.get('name'),str) or case.get('status') not in allowed_states or not isinstance(case.get('metrics'),list) or not isinstance(case.get('settings',{}),dict):raise ValueError('Invalid case')
                for m in case.get('metrics',[]):
                    if not isinstance(m,dict) or not isinstance(m.get('name'),str) or not isinstance(m.get('unit'),str) or not isinstance(m.get('value'),(float,int)) or not math.isfinite(m['value']) or m.get('direction') not in ('higher','lower','context'):
                        raise ValueError('Invalid metric')
        cfg=dict(value['config']);profile=cfg.pop('profile','quick');config(profile,**cfg)
        required = ("snapshot", "config", "results")
        expected = fingerprint({k: value[k] for k in required})
        if expected != value.get("bundle_fingerprint"):
            raise ValueError("Bundle content fingerprint mismatch")
        rid = self.create(value["metadata"], value["snapshot"], value["config"], value["tests"], value["gpu"], value.get("simulated", False))
        for result in value["results"]:
            if not isinstance(result, dict) or "test_id" not in result or "status" not in result:
                raise ValueError("Invalid result record")
            self.result(rid, result)
        for sample in value.get("telemetry", []):
            data = {k: v for k, v in sample.items() if k not in ("ts", "test_id")}
            self.sample(rid, sample.get("test_id"), sample["ts"], data)
        status = value.get("status", "imported")
        if status in ACTIVE:
            status = "interrupted"
        with self.connect() as db:
            db.execute("UPDATE runs SET status=?,source='import',created=? WHERE id=?", (status, value["created"], rid))
        self.audit(rid, dict(action="imported", original_id=value["id"], original_audit=value.get("audit", [])))
        return rid
