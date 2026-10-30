"""Browser maintenance controller for installer-managed user services."""
import fcntl
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from . import __version__
from .updater import BUSY, git, redact, stamp, write_json


class MaintenanceConflict(ValueError):
    pass


@contextmanager
def launch_guard(store):
    """Serialize CLI/browser launches with the maintenance request handoff."""
    root = store.root / "maintenance"
    root.mkdir(exist_ok=True)
    with (root / "request.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "status.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["state"] in BUSY and record.get("action") in ("update", "restart"):
                raise MaintenanceConflict("Service maintenance is in progress. New benchmarks are paused until it finishes.")
        yield


class Updates:
    def __init__(self, store, instance, health_url):
        self.store = store
        self.instance = instance
        self.health_url = health_url
        self.lock = threading.RLock()
        self.root = store.root / "maintenance"
        self.root.mkdir(exist_ok=True)
        self.status_path = self.root / "status.json"
        self.config_path = Path(os.environ["GPU_LAB_DEPLOYMENT"]) if os.environ.get("GPU_LAB_DEPLOYMENT") else None
        self.config = {}
        self.config_error = None
        if self.config_path:
            try:
                self.config = json.loads(self.config_path.read_text())
            except (OSError, ValueError) as error:
                self.config_error = f"Cannot read deployment configuration: {error}"

    def record(self):
        try:
            return json.loads(self.status_path.read_text())
        except FileNotFoundError:
            return dict(state="idle", message="No maintenance has run yet.")

    def unit_active(self, unit):
        result = subprocess.run(["systemctl", "--user", "show", unit, "--property=ActiveState", "--value"],
                                capture_output=True, text=True, timeout=5)
        return result.returncode == 0 and result.stdout.strip() in ("active", "activating", "reloading", "deactivating")

    def reason(self):
        if self.config_error:
            return self.config_error
        if not self.config:
            return "Browser updates need one-time setup on the Linux server. Run the installer with --update-app --service."
        if not self.config.get("service"):
            return "This installation is not configured as a user service. Run the installer with --update-app --service."
        if not all(shutil.which(tool) for tool in ("git", "systemctl", "systemd-run")):
            return "Browser updates require Linux with Git and user systemd services."
        if not Path(self.config.get("python", "")).is_file():
            return "The installed application Python interpreter is missing. Rerun the installer."
        try:
            git(self.config["source"], "rev-parse", "--show-toplevel", timeout=5)
        except (KeyError, RuntimeError, OSError, subprocess.TimeoutExpired):
            return "The configured source is not a Git checkout. Install from a Git clone to enable browser updates."
        if not self.unit_active(self.config["service"]):
            return "The GPU Lab user service is not running. Start it through the installer with --update-app --service."
        if os.environ.get("SYSTEMD_EXEC_PID") != str(os.getpid()):
            return "This browser is connected to a manually launched app. Connect to the installer-managed GPU Lab service for maintenance."
        return None

    def ensure_idle(self):
        record = self.record()
        if record["state"] in BUSY and record.get("action") in ("update", "restart"):
            raise MaintenanceConflict("A service update or restart is in progress. New benchmarks are paused until it finishes.")

    def status(self):
        with self.lock:
            record = self.record()
            if record["state"] in BUSY and time.time() - record.get("started_unix", time.time()) > 30:
                try:
                    running = self.unit_active(record["unit"])
                except (OSError, subprocess.TimeoutExpired):
                    running = False
                if not running:
                    record.update(state="failed", message="The maintenance worker stopped unexpectedly. Review the update log before retrying.", updated=stamp())
                    write_json(self.status_path, record)
            reason = self.reason()
            available = record.get("available_commit")
            log = ""
            if record.get("id"):
                path = self.root / record["id"] / "update.log"
                if path.exists():
                    with path.open("rb") as stream:
                        stream.seek(max(0, path.stat().st_size - 64000))
                        log = stream.read().decode(errors="replace")
            return dict(version=__version__, instance=self.instance, installed_commit=self.config.get("installed_commit"),
                        branch=self.config.get("branch", "main"), supported=reason is None, unavailable_reason=reason,
                        installation_uncertain=self.config.get("installation_uncertain", False),
                        update_available=bool(available and available != self.config.get("installed_commit")),
                        busy=record["state"] in BUSY, operation=record, log=redact(log))

    def start(self, action):
        if action not in ("check", "update", "restart"):
            raise ValueError("Unknown maintenance action")
        with self.lock, (self.root / "request.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            # Reconcile a worker interrupted by a host reboot before accepting work.
            status = self.status()
            if status["busy"]:
                raise MaintenanceConflict("Another maintenance operation is already in progress.")
            if not status["supported"]:
                raise ValueError(status["unavailable_reason"])
            identity = uuid.uuid4().hex
            folder = self.root / identity
            folder.mkdir(mode=0o700)
            worker = folder / "updater.py"
            shutil.copyfile(Path(__file__).with_name("updater.py"), worker)
            record = {key: value for key, value in status["operation"].items() if key in ("available_commit", "checked_at")}
            record.update(id=identity, action=action, state="starting", message="Starting background maintenance.",
                          started=stamp(), updated=stamp(), started_unix=time.time(), unit="gpu-lab-maintenance-" + identity)
            job = dict(action=action, config_path=str(self.config_path), data=str(self.store.root.resolve()),
                       status_path=str(self.status_path.resolve()), log_path=str(folder.resolve() / "update.log"),
                       instance=self.instance, health_url=self.health_url)
            job_path = folder / "job.json"
            write_json(job_path, job)
            write_json(folder / "status.json", record)
            write_json(self.status_path, record)
            try:
                result = subprocess.run(["systemd-run", "--user", "--quiet", "--collect", "--property=Type=exec",
                                         "--unit=" + record["unit"], "--", self.config["updater_python"], str(worker.resolve()), str(job_path.resolve())],
                                        capture_output=True, text=True, timeout=15)
                if result.returncode:
                    raise RuntimeError(redact(result.stderr.strip() or "Could not launch maintenance worker"))
            except Exception as error:
                # A timeout may mean systemd accepted the job; avoid permitting a duplicate.
                try:
                    running = self.unit_active(record["unit"])
                except (OSError, subprocess.TimeoutExpired):
                    running = False
                if not running:
                    record.update(state="failed", message=str(error), updated=stamp())
                    write_json(self.status_path, record)
                    raise ValueError(str(error)) from error
            return record
