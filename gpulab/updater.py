"""Standalone, stdlib-only maintenance worker; copied before changing the app.

Run by a separate user systemd unit using the base Python interpreter, so stopping
GPU Lab or replacing its virtualenv packages cannot stop this worker.
"""
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

BUSY = {"starting", "checking", "waiting", "downloading", "stopping", "backing_up", "installing", "starting_service", "verifying"}


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def redact(text):
    return re.sub(r"(https?://)[^/\s@]+@", r"\1[redacted]@", text)


def git(source, *args, timeout=120):
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_SSH_COMMAND": "ssh -oBatchMode=yes -oConnectTimeout=15"}
    result = subprocess.run(["git", "-C", str(source), *args], capture_output=True, text=True, timeout=timeout, env=env)
    if result.returncode:
        raise RuntimeError(redact(result.stderr.strip() or result.stdout.strip() or "Git command failed"))
    return result.stdout.strip()


def clean_head(source):
    if git(source, "status", "--porcelain", "--untracked-files=normal"):
        raise RuntimeError("The server checkout has local changes. Commit or remove those changes before updating.")
    return git(source, "rev-parse", "HEAD")


def fetch_target(config):
    source = config["source"]
    branch = config["branch"]
    git(source, "check-ref-format", "--branch", branch)
    git(source, "fetch", "--no-tags", "origin", "refs/heads/" + branch)
    target = git(source, "rev-parse", "FETCH_HEAD")
    head = git(source, "rev-parse", "HEAD")
    # Validate before stopping the app; merge --ff-only is also enforced later.
    try:
        git(source, "merge-base", "--is-ancestor", head, target)
    except RuntimeError as error:
        raise RuntimeError("The server checkout cannot fast-forward to the configured GitHub branch. Resolve local/divergent history in the terminal before updating.") from error
    return target


def active_runs(data):
    """Only local live workers block maintenance; dead workers need no browser."""
    with closing(sqlite3.connect(Path(data) / "lab.sqlite3", timeout=30)) as db:
        rows = db.execute("SELECT pid,marker,created FROM runs WHERE source='local' AND (pid IS NOT NULL OR status IN ('queued','preflight','running','cancelling'))").fetchall()
    count = 0
    for pid, marker, created in rows:
        if not pid:
            if time.time() - datetime.fromisoformat(created).timestamp() < 60:
                count += 1
            continue
        try:
            stat = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()
            current = Path("/proc/sys/kernel/random/boot_id").read_text().strip() + ":" + stat[19]
            if current == marker:
                count += 1
        except (OSError, IndexError):
            pass
    return count


class Worker:
    def __init__(self, job):
        self.job = job
        self.config_path = Path(job["config_path"])
        self.config = json.loads(self.config_path.read_text())
        self.status_path = Path(job["status_path"])
        self.record = json.loads(self.status_path.read_text())
        self.log_path = Path(job["log_path"])

    def log(self, message):
        with self.log_path.open("a") as log:
            log.write(f"{stamp()} {redact(message)}\n")

    def phase(self, state, message, **fields):
        self.record.update(state=state, message=message, updated=stamp(), **fields)
        write_json(self.status_path, self.record)
        write_json(self.log_path.parent / "status.json", self.record)
        self.log(message)

    def command(self, args, timeout=120):
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                                env={**os.environ, "PIP_NO_INPUT": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"})
        if result.stdout:
            self.log(result.stdout[-64000:])
        if result.stderr:
            self.log(result.stderr[-64000:])
        if result.returncode:
            raise RuntimeError(f"{Path(args[0]).name} failed (exit {result.returncode}). See the update log.")

    def service(self, action):
        self.command(["systemctl", "--user", action, self.config["service"]])

    def wait_for_runs(self):
        while True:
            count = active_runs(self.job["data"])
            if not count:
                return
            if self.record.get("active_runs") != count:
                self.phase("waiting", f"Waiting for {count} active benchmark(s) to finish. New launches are paused.", active_runs=count)
            time.sleep(2)

    def backup(self):
        destination = self.log_path.parent / "lab.sqlite3.backup"
        with closing(sqlite3.connect(Path(self.job["data"]) / "lab.sqlite3", timeout=30)) as source:
            with closing(sqlite3.connect(destination)) as target:
                source.backup(target)
        self.record["backup"] = str(destination)

    def verify(self, expected_commit):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                with urlopen(self.job["health_url"], timeout=3) as response:
                    health = json.load(response)
                if (health.get("ok") and health.get("instance") != self.job["instance"]
                        and health.get("commit") == expected_commit):
                    return
            except (OSError, ValueError):
                pass
            time.sleep(1)
        raise RuntimeError("The service did not return healthy within 60 seconds. Check the update log and user service journal.")

    def run(self):
        stopped = False
        installing = False
        try:
            action = self.job["action"]
            target = self.config.get("installed_commit")
            if action in ("check", "update"):
                self.phase("checking", "Checking GitHub for the latest code.")
                if action == "update":
                    clean_head(self.config["source"])
                target = fetch_target(self.config)
                self.record.update(available_commit=target, checked_at=stamp())
                if action == "check":
                    self.phase("completed", "Update check complete.")
                    return
                if target == self.config.get("installed_commit") and not self.config.get("installation_uncertain"):
                    self.phase("completed", "GPU Lab is already up to date.")
                    return
            self.wait_for_runs()
            if action == "update":
                previous = clean_head(self.config["source"])
                self.phase("downloading", "Applying the downloaded code.", previous_commit=previous, target_commit=target, active_runs=0)
                # FETCH_HEAD may have changed outside the app; use the exact checked SHA.
                git(self.config["source"], "merge", "--ff-only", target)
            self.phase("stopping", "Stopping GPU Lab. The browser will reconnect automatically.")
            # Also attempt a start if stop times out after systemd accepted the request.
            stopped = True
            self.service("stop")
            if action == "update":
                self.phase("backing_up", "Backing up benchmark history before installation.")
                self.backup()
                self.phase("installing", "Installing the updated application and its dependencies.")
                installing = True
                package = self.config["source"] + ("[opencl]" if self.config.get("opencl") else "")
                self.command([self.config["python"], "-m", "pip", "install", "--upgrade", "--upgrade-strategy", "only-if-needed", package], timeout=900)
                self.config.update(installed_commit=target, installation_uncertain=False)
                write_json(self.config_path, self.config)
                installing = False
            self.phase("starting_service", "Starting GPU Lab.")
            self.service("start")
            stopped = False
            self.phase("verifying", "Checking that GPU Lab restarted successfully.")
            self.verify(target)
            self.phase("completed", "Update installed successfully." if action == "update" else "Service restarted successfully.")
        except Exception as error:
            if installing:
                self.config["installation_uncertain"] = True
                write_json(self.config_path, self.config)
            if stopped:
                try:
                    self.log("Attempting to start the service after the failed update.")
                    self.service("start")
                except Exception as restart_error:
                    self.log(str(restart_error))
            self.phase("failed", redact(str(error)))
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(Worker(json.loads(Path(sys.argv[1]).read_text())).run())
