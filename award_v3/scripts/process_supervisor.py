#!/usr/bin/env python3
"""Linux subreaperでGazeboの独立した子孫プロセスも回収する。追加依存なし。"""
import argparse
import ctypes
from datetime import datetime
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def enable_subreaper():
    # Gazeboがfork/setsidしても、親終了後のプロセスを本supervisorが引き取る。
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'subreaper could not be enabled')


def descendants():
    parents = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            parents[int(entry.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    found = {os.getpid()}
    while True:
        added = {pid for pid, parent in parents.items() if parent in found} - found
        if not added:
            break
        found.update(added)
    return found - {os.getpid()}


def signal_pids(pids, sig):
    for pid in pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def reap():
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                return
        except ChildProcessError:
            return


def cleanup(children, controller):
    # 先に制御ノードへ停止指令送信・記録保存の猶予を与える。
    if controller is not None and controller.poll() is None:
        controller.send_signal(signal.SIGINT)
        try:
            controller.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
    # 固定PIDや他のGazeboを対象にしない。本supervisorの子孫だけを終了。
    for child in children:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        pending = descendants()
        if not pending:
            break
        signal_pids(pending, signal.SIGTERM)
        reap()
        time.sleep(.1)
    signal_pids(descendants(), signal.SIGKILL)
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        reap()
        if not descendants():
            break
        time.sleep(.05)


def run(root, case, mode, action):
    lock = (root / '.run.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        print('発展版がすでに動いています。AI wall patrol V3でqを押してから再実行してください。', flush=True)
        return 2
    children, handles = [], []
    controller = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    old_handlers = {sig: signal.signal(sig, interrupted)
                    for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        enable_subreaper()
        log_dir = root / 'logs' / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + case)
        log_dir.mkdir(parents=True)
        print(f'Launching {case} / {mode}. Logs: {log_dir}', flush=True)
        def launch(command, filename=None):
            handle = None
            if filename:
                handle = (log_dir / filename).open('w')
                handles.append(handle)
            child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                     start_new_session=True, close_fds=True, cwd=root)
            children.append(child)
            return child
        launch(['ign', 'gazebo', '-r', str(root / 'worlds' / (case + '.sdf'))], 'gazebo.log')
        launch(['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
                '/model/vehicle_blue/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',
                '/wall/camera@sensor_msgs/msg/Image[ignition.msgs.Image',
                '/model/vehicle_blue/odometry@nav_msgs/msg/Odometry[ignition.msgs.Odometry'], 'bridge.log')
        command = [sys.executable, '-u', str(root / 'scripts/patrol_v3.py'),
                   '--scenario', case, '--mode', mode]
        if action in ('auto','capture'):
            command.append('--autostart')
        if action == 'capture':
            command.append('--capture-only')
        controller = launch(command)
        # 起動時に入力を待ち続ける場合も回収する。手動モードの操作時間は制限しない。
        deadline = time.monotonic() + 240 if action in ('auto','capture') else None
        while controller.poll() is None:
            if deadline is not None and time.monotonic() > deadline:
                print('自動走行の待機時間を超えたため終了します。', flush=True)
                return 124
            time.sleep(.1)
        return controller.returncode
    except KeyboardInterrupt:
        return 130
    finally:
        # 終了中の追加Ctrl+Cで回収処理が中断されないようにする。
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            cleanup(children, controller)
        finally:
            for handle in handles:
                handle.close()
            lock.close()
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        print('Gazebo・ブリッジの終了処理が完了しました。', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', choices=['CASE_A', 'CASE_B', 'CASE_C', 'CASE_D'])
    parser.add_argument('mode', choices=['cnn'])
    parser.add_argument('action', choices=['manual', 'auto', 'capture'], default='manual', nargs='?')
    args = parser.parse_args()
    return run(ROOT, args.case, args.mode, args.action)


if __name__ == '__main__':
    sys.exit(main())
