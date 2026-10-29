"""Gazebo代替プロセスで、別セッション化したサーバーの回収を検証する。"""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/process_supervisor.py'


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name in ('scripts', 'bin', 'worlds'):
            (self.root / name).mkdir()
        shutil.copy2(SOURCE, self.root / 'scripts/process_supervisor.py')
        fake = '''#!/usr/bin/env python3
import os, signal, sys, time
from pathlib import Path
name=Path(sys.argv[0]).name
if os.fork():
    sys.exit(0)
os.setsid()
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(name+'.pid').write_text(str(os.getpid()))
while True: time.sleep(.1)
'''
        for name in ('ign', 'ros2'):
            p = self.root / 'bin' / name
            p.write_text(fake)
            p.chmod(0o755)
        (self.root / 'scripts/patrol_v3.py').write_text(
            'import os,time\ntime.sleep(float(os.environ.get("TEST_DURATION",".8")))\n')
        self.env = dict(os.environ, PATH=str(self.root / 'bin') + ':' + os.environ['PATH'])
        self.processes = []

    def tearDown(self):
        for proc in self.processes:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=10)
        # テストが失敗した場合も代替サーバーを残さない。
        for file in self.root.glob('*.pid'):
            try:
                os.kill(int(file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.temp.cleanup()

    def launch(self, duration='.8'):
        env = dict(self.env, TEST_DURATION=duration)
        proc = subprocess.Popen([sys.executable, '-u', str(self.root / 'scripts/process_supervisor.py'),
                                 'CASE_A', 'cnn', 'auto'], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.processes.append(proc)
        return proc

    def ready(self):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if all((self.root / (name + '.pid')).exists() for name in ('ign', 'ros2')):
                return [int((self.root / (name + '.pid')).read_text()) for name in ('ign', 'ros2')]
            time.sleep(.02)
        self.fail('fake servers did not start')

    def assert_gone(self, pids):
        for pid in pids:
            self.assertFalse(Path('/proc', str(pid)).exists(), f'orphan process left: {pid}')

    def test_detached_servers_reaped_and_second_run_can_start(self):
        first = self.launch()
        pids = self.ready()
        output = first.communicate(timeout=12)[0]
        self.assertEqual(first.returncode, 0, output)
        self.assert_gone(pids)
        for path in self.root.glob('*.pid'):
            path.unlink()
        second = self.launch()
        pids = self.ready()
        output = second.communicate(timeout=12)[0]
        self.assertEqual(second.returncode, 0, output)
        self.assert_gone(pids)

    def test_active_lock_denies_second_launcher(self):
        first = self.launch('30')
        pids = self.ready()
        second = self.launch()
        output = second.communicate(timeout=5)[0]
        self.assertEqual(second.returncode, 2, output)
        self.assertIn('すでに動いています', output)
        first.send_signal(signal.SIGINT)
        first.communicate(timeout=12)
        self.assertEqual(first.returncode, 130)
        self.assert_gone(pids)


if __name__ == '__main__':
    unittest.main()
