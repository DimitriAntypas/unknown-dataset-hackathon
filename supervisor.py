"""Portable CLI wall-clock supervision; no scientific imports in the parent."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def run(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--out-dir', default='out')
    parser.add_argument('--time-budget', type=float, default=40)
    opts, _ = parser.parse_known_args(argv)
    if not math.isfinite(opts.time_budget) or opts.time_budget <= 0:
        parser.error('--time-budget must be finite and positive')
    if '--help' in argv or '-h' in argv:
        return subprocess.call([sys.executable, str(Path(__file__).with_name('main.py')), '--_worker', *argv])
    started = time.monotonic()
    # Reserve at least eight seconds of the contest's 60-second allowance for
    # parent startup, termination and publishing; this is not a hardware guarantee.
    deadline = min(52.0, opts.time_budget + 8.0)
    out = Path(opts.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='udh-worker-') as scratch:
        env = os.environ.copy()
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            env[key] = '1'
        command = [sys.executable, str(Path(__file__).with_name('main.py')), '--_worker', *argv, '--out-dir', scratch]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=deadline)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            stdout, stderr = process.communicate()
        root = Path(scratch)
        if timed_out:
            marker = root / 'checkpoint.json'
            if not marker.exists():
                print('Time limit reached before a usable checkpoint was ready.', file=sys.stderr)
                return 124
            source = root / json.loads(marker.read_text())['directory']
        elif process.returncode:
            print(stderr or stdout, file=sys.stderr)
            return process.returncode
        else:
            source = root
        profile = json.loads((source / 'profile.json').read_text())
        report = profile['model_report']
        report['supervisor_timeout'] = timed_out
        report['wall_seconds'] = round(time.monotonic() - started, 3)
        for name in ('predictions.csv', 'profile.json'):
            temporary = out / (name + '.tmp')
            if name == 'profile.json':
                temporary.write_text(json.dumps(profile, indent=2))
            else:
                shutil.copyfile(source / name, temporary)
            os.replace(temporary, out / name)
        print(json.dumps({'task': profile['task'], 'wall_seconds': report['wall_seconds'],
                          'timeout_fallback': timed_out, 'models': report.get('chosen_models')}))
        return 0
