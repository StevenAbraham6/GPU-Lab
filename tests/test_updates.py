import http.client
import json
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from gpulab.runner import create_run
from gpulab.server import LabServer
from gpulab.store import Store
from gpulab.updater import Worker, active_runs, clean_head, git, write_json
from gpulab.updates import MaintenanceConflict, Updates, launch_guard


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = Store(self.root / "data")
        self.remote = self.root / "remote"
        subprocess.run(["git", "init", "--initial-branch=main", str(self.remote)], check=True, capture_output=True)
        git(self.remote, "config", "user.name", "Update Test")
        git(self.remote, "config", "user.email", "test@example.test")
        (self.remote / "app.txt").write_text("version one\n")
        git(self.remote, "add", ".")
        git(self.remote, "commit", "-m", "Initial version")
        self.source = self.root / "source"
        subprocess.run(["git", "clone", str(self.remote), str(self.source)], check=True, capture_output=True)
        self.initial = git(self.source, "rev-parse", "HEAD")
        (self.remote / "app.txt").write_text("version two\n")
        git(self.remote, "commit", "-am", "New version")
        self.latest = git(self.remote, "rev-parse", "HEAD")
        self.config_path = self.root / "deployment.json"
        self.config = dict(source=str(self.source), python=sys.executable, updater_python=sys.executable,
                           branch="main", service="gpu-lab.service", installed_commit=self.initial, opencl=True)
        write_json(self.config_path, self.config)
        self.environment = patch.dict(os.environ, {"GPU_LAB_DEPLOYMENT": str(self.config_path)})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.controller = Updates(self.store, "old-instance", "http://127.0.0.1:8080/api/health")

    def worker(self, action="update"):
        folder = self.controller.root / "test-job"
        folder.mkdir(exist_ok=True)
        write_json(self.controller.status_path, dict(id="test-job", action=action, state="starting"))
        return Worker(dict(action=action, config_path=str(self.config_path), data=str(self.store.root),
                           status_path=str(self.controller.status_path), log_path=str(folder / "update.log"),
                           instance="old-instance", health_url=self.controller.health_url))

    def record(self):
        return json.loads(self.controller.status_path.read_text())

    def test_check_fetches_without_changing_checkout_or_installing(self):
        worker = self.worker("check")
        with patch.object(worker, "command") as command:
            worker.run()
        command.assert_not_called()
        self.assertEqual(git(self.source, "rev-parse", "HEAD"), self.initial)
        self.assertEqual(self.record()["available_commit"], self.latest)
        self.assertEqual(self.record()["state"], "completed")

    def test_update_waits_then_stops_backs_up_installs_starts_and_verifies(self):
        # Detached HEAD is produced by the bootstrap installer too.
        git(self.source, "checkout", "--detach", self.initial)
        worker = self.worker()
        events = []
        backup = worker.backup

        def command(args, **kwargs):
            if args[0] == "systemctl":
                events.append(args[2])
            else:
                events.append("install")
                self.assertEqual(events, ["wait", "stop", "backup", "install"])
                self.assertIn(str(self.source) + "[opencl]", args)
                self.assertTrue((worker.log_path.parent / "lab.sqlite3.backup").exists())

        def do_backup():
            events.append("backup")
            backup()

        with patch("gpulab.updater.active_runs", side_effect=[1, 0]), \
                patch("gpulab.updater.time.sleep", side_effect=lambda _: events.append("wait")), \
                patch.object(worker, "backup", side_effect=do_backup), \
                patch.object(worker, "command", side_effect=command), \
                patch.object(worker, "verify", side_effect=lambda commit: events.append("verify")):
            self.assertEqual(worker.run(), 0)
        self.assertEqual(events, ["wait", "stop", "backup", "install", "start", "verify"])
        self.assertEqual(git(self.source, "rev-parse", "HEAD"), self.latest)
        self.assertEqual(json.loads(self.config_path.read_text())["installed_commit"], self.latest)
        self.assertEqual(self.record()["previous_commit"], self.initial)
        self.assertEqual(self.record()["state"], "completed")
        with sqlite3.connect(self.record()["backup"]) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM runs").fetchone()[0], 0)

    def test_dirty_checkout_fails_before_stopping_service(self):
        (self.source / "app.txt").write_text("local edits\n")
        worker = self.worker()
        with patch.object(worker, "service") as service:
            self.assertEqual(worker.run(), 1)
        service.assert_not_called()
        self.assertIn("local changes", self.record()["message"])
        self.assertEqual((self.source / "app.txt").read_text(), "local edits\n")

    def test_untracked_files_are_preserved_and_block_updates(self):
        (self.source / "important.txt").write_text("keep me")
        with self.assertRaisesRegex(RuntimeError, "local changes"):
            clean_head(self.source)

    def test_diverged_history_fails_before_stopping_service(self):
        git(self.source, "config", "user.name", "Local")
        git(self.source, "config", "user.email", "local@example.test")
        (self.source / "local.txt").write_text("different branch")
        git(self.source, "add", ".")
        git(self.source, "commit", "-m", "Local commit")
        worker = self.worker()
        with patch.object(worker, "service") as service:
            self.assertEqual(worker.run(), 1)
        service.assert_not_called()
        self.assertEqual(self.record()["state"], "failed")

    def test_failed_install_attempts_start_and_can_retry_same_commit(self):
        worker = self.worker()
        events = []

        def command(args, **kwargs):
            if args[0] == "systemctl":
                events.append(args[2])
            else:
                raise RuntimeError("Simulated dependency installation failure")

        with patch.object(worker, "command", side_effect=command):
            self.assertEqual(worker.run(), 1)
        self.assertEqual(events, ["stop", "start"])
        config = json.loads(self.config_path.read_text())
        self.assertTrue(config["installation_uncertain"])
        self.assertEqual(config["installed_commit"], self.initial)
        self.assertEqual(git(self.source, "rev-parse", "HEAD"), self.latest)
        retry = self.worker()
        with patch.object(retry, "command"), patch.object(retry, "verify"):
            self.assertEqual(retry.run(), 0)
        self.assertFalse(json.loads(self.config_path.read_text())["installation_uncertain"])
        self.assertEqual(self.record()["state"], "completed")

    def test_restart_never_fetches_or_installs(self):
        worker = self.worker("restart")
        with patch("gpulab.updater.fetch_target") as fetch, patch.object(worker, "command") as command, \
                patch.object(worker, "verify") as verify:
            self.assertEqual(worker.run(), 0)
        fetch.assert_not_called()
        self.assertEqual([call.args[0][2] for call in command.call_args_list], ["stop", "start"])
        verify.assert_called_once_with(self.initial)

    def test_health_failure_is_recorded(self):
        worker = self.worker("restart")
        with patch.object(worker, "command"), patch.object(worker, "verify", side_effect=RuntimeError("Not healthy")):
            self.assertEqual(worker.run(), 1)
        self.assertEqual(self.record()["state"], "failed")
        self.assertIn("Not healthy", self.record()["message"])

    def test_verify_requires_new_instance_and_expected_commit(self):
        worker = self.worker()
        responses = []
        for instance, commit in [("old-instance", self.latest), ("new-instance", self.initial), ("new-instance", self.latest)]:
            response = Mock()
            response.read.return_value = json.dumps(dict(ok=True, instance=instance, commit=commit)).encode()
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            responses.append(response)
        with patch("gpulab.updater.urlopen", side_effect=responses) as request, patch("gpulab.updater.time.sleep"):
            worker.verify(self.latest)
        self.assertEqual(request.call_count, 3)

    def test_network_listener_maintenance_health_check_uses_loopback(self):
        server = LabServer(("0.0.0.0", 0), self.store, token="secret")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            worker = self.worker("restart")
            worker.job["health_url"] = server.updates.health_url
            worker.verify(self.initial)
            self.assertTrue(server.updates.health_url.startswith("http://127.0.0.1:"))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_up_to_date_update_does_not_stop_service(self):
        self.config["installed_commit"] = self.latest
        write_json(self.config_path, self.config)
        worker = self.worker()
        with patch.object(worker, "command") as command:
            worker.run()
        command.assert_not_called()
        self.assertIn("already up to date", self.record()["message"])

    def test_background_handoff_copies_worker_and_prevents_duplicate_jobs(self):
        with patch.object(self.controller, "reason", return_value=None), \
                patch("gpulab.updates.subprocess.run", return_value=Mock(returncode=0)) as run:
            record = self.controller.start("update")
            with self.assertRaises(MaintenanceConflict):
                self.controller.start("restart")
        folder = self.controller.root / record["id"]
        self.assertEqual((folder / "updater.py").read_bytes(), Path("gpulab/updater.py").read_bytes())
        self.assertIn("--user", run.call_args.args[0])
        self.assertEqual(json.loads((folder / "job.json").read_text())["action"], "update")

    def test_failed_launch_releases_maintenance_and_reports_error(self):
        with patch.object(self.controller, "reason", return_value=None), \
                patch.object(self.controller, "unit_active", return_value=False), \
                patch("gpulab.updates.subprocess.run", return_value=Mock(returncode=1, stderr="Cannot connect to user bus")):
            with self.assertRaisesRegex(ValueError, "user bus"):
                self.controller.start("update")
        self.assertEqual(self.record()["state"], "failed")
        with launch_guard(self.store):
            pass

    def test_interrupted_worker_is_reconciled(self):
        write_json(self.controller.status_path, dict(id="test-job", action="update", state="installing", started_unix=time.time()-90, unit="gone"))
        with patch.object(self.controller, "unit_active", return_value=False), patch.object(self.controller, "reason", return_value=None):
            self.assertFalse(self.controller.status()["busy"])
        self.assertEqual(self.record()["state"], "failed")

    def test_new_runs_blocked_during_updates_but_not_checks(self):
        self.worker("update")
        with self.assertRaises(MaintenanceConflict):
            create_run(self.store, names=["latency"], simulated=True)
        self.assertEqual(self.store.list(), [])
        self.worker("check")
        with launch_guard(self.store):
            pass

    def test_auth_origin_guards_and_unsupported_installation(self):
        with patch.dict(os.environ, {"GPU_LAB_DEPLOYMENT": ""}):
            server = LabServer(("127.0.0.1", 0), self.store, token="secret")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def request(method, path, headers=None):
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
            connection.request(method, path, "{}" if method == "POST" else None,
                               {"Content-Type": "application/json", **(headers or {})})
            response = connection.getresponse()
            payload = response.read()
            connection.close()
            return response.status, payload

        try:
            self.assertEqual(request("GET", "/api/health")[0], 200)
            self.assertEqual(request("GET", "/api/settings")[0], 401)
            self.assertEqual(request("POST", "/api/settings/update")[0], 401)
            auth = {"Authorization": "Bearer secret"}
            self.assertEqual(request("POST", "/api/settings/update", {**auth, "Origin": "https://evil.test"})[0], 403)
            code, body = request("GET", "/api/settings", auth)
            self.assertEqual(code, 200)
            self.assertFalse(json.loads(body)["supported"])
            self.assertEqual(request("POST", "/api/settings/update", auth)[0], 400)
            self.worker()
            self.assertEqual(request("POST", "/api/runs", auth)[0], 409)
            self.assertEqual(request("GET", "/settings.js")[0], 200)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_dead_and_imported_workers_do_not_block_updates(self):
        rid = self.store.create(dict(name="Dead worker"), {}, {}, [], {}, False)
        self.store.update(rid, pid=999999999, marker="missing")
        self.assertEqual(active_runs(self.store.root), 0)

    def test_installer_writes_deployment_and_launcher_without_gpu_reinstallation(self):
        # Mock package downloads, but execute the installer's actual configuration
        # generator and launcher. No user service or home-directory files change.
        shutil.copyfile("install.sh", self.source / "install.sh")
        (self.source / "pyproject.toml").write_text("[project]\nname='installer-test'\n")
        interpreter = self.root / "installer-python"
        interpreter.write_text('''#!/usr/bin/env bash
set -eu
if [[ "$1" == '-I' ]]; then shift;fi
if [[ "$1" == '-m' && "$2" == 'venv' ]]; then
  mkdir -p "$3/bin"
  cp "$0" "$3/bin/python"
  exit 0
fi
if [[ "$1" == '-m' && "$2" == 'pip' ]]; then
  printf '%s\\n' "$*" >> "$GPU_LAB_TEST_LOG"
  exit 0
fi
if [[ "$1" == '-c' && "$2" == *'find_spec("pyopencl")'* ]]; then exit 0;fi
if [[ "$1" == '-m' && "$2" == 'gpulab' ]]; then
  printf '%s\\n' "$PWD" >> "$GPU_LAB_TEST_LOG"
  exit 0
fi
exec ''' + shlex.quote(sys.executable) + ''' "$@"
''')
        interpreter.chmod(0o755)
        prefix = self.root / "application with spaces"
        binaries = self.root / "bin with spaces"
        log = self.root / "install.log"
        result = subprocess.run(["bash", str(self.source / "install.sh"), "--app-only", "--update-app"],
                                capture_output=True, text=True, timeout=30,
                                env={**os.environ, "GPU_LAB_PREFIX": str(prefix), "GPU_LAB_HOME": str(self.store.root),
                                     "GPU_LAB_BIN": str(binaries), "GPU_LAB_PYTHON": str(interpreter), "GPU_LAB_TEST_LOG": str(log)})
        self.assertEqual(result.returncode, 0, result.stderr)
        deployment = json.loads((prefix / "deployment.json").read_text())
        self.assertEqual(deployment["source"], str(self.source.resolve()))
        self.assertEqual(deployment["python"], str(prefix / "venv/bin/python"))
        self.assertEqual(deployment["installed_commit"], self.initial)
        self.assertTrue(deployment["opencl"])
        self.assertEqual(deployment["branch"], "main")
        self.assertIsNone(deployment["service"])
        self.assertIn("GPU_LAB_DEPLOYMENT=", (binaries / "gpu-lab").read_text())
        self.assertIn(' -I -m gpulab', (binaries / "gpu-lab").read_text())
        self.assertNotIn("torch", log.read_text())

        # Exercise the real systemd unit and drop-in generation with a fake manager.
        fake_bin = self.root / "fake-bin"
        fake_bin.mkdir()
        systemctl = fake_bin / "systemctl"
        systemctl.write_text("#!/bin/sh\nexit 0\n")
        systemctl.chmod(0o755)
        home = self.root / "home"
        home.mkdir()
        env = {**os.environ, "HOME": str(home), "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
               "GPU_LAB_PREFIX": str(prefix), "GPU_LAB_HOME": str(self.store.root), "GPU_LAB_BIN": str(binaries),
               "GPU_LAB_PYTHON": str(interpreter), "GPU_LAB_TEST_LOG": str(log)}
        args = ["bash", str(self.source / "install.sh"), "--app-only", "--update-app"]
        result = subprocess.run(args + ["--network-service"], capture_output=True, text=True, timeout=30, env=env)
        self.assertEqual(result.returncode, 0, result.stderr)
        token = prefix / "network-token"
        original = token.read_text()
        self.assertGreaterEqual(len(original.strip()), 32)
        self.assertEqual(token.stat().st_mode & 0o777, 0o600)
        override = home / ".config/systemd/user/gpu-lab.service.d/network.conf"
        config = override.read_text()
        self.assertIn("--host 0.0.0.0 --port 8080 --token-file", config)
        self.assertIn(str(token), config)
        for option in ("--service", "--network-service"):
            result = subprocess.run(args + [option], capture_output=True, text=True, timeout=30, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(override.read_text(), config)
            self.assertEqual(token.read_text(), original)


if __name__ == "__main__":
    unittest.main()
