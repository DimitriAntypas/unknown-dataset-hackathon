"""Regression checks for the reviewed defects; runnable without pytest."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main as M


def test_fresh_budget():
    old = M.Budget(10)
    old.started -= 100
    assert old.left() < 0
    assert M.Budget(10).left() > 9


def test_constant_clusters_and_ids():
    df = pd.DataFrame({'id': np.arange(101, 131), 'constant': 1})
    r = M.solve(df, task='clustering', n_clusters=3, id_col='id')
    assert len(set(r['labels'])) == 3
    assert np.array_equal(r['ids'], df.id)
    assert r['id_name'] == 'id'
    assert len(set(M.solve(df, task='clustering', n_clusters=1)['labels'])) == 1
    try:
        M.solve(df, task='clustering', n_clusters=31)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid k must fail')


def test_missing_rows():
    df = pd.DataFrame({'a':[1.,np.nan,2.,np.nan], 'b':['x','?',' ','y']})
    cleaner = M.FeatureCleaner().fit(df)
    p = M.build_profile(df, cleaner, None, 'clustering', 2, None, {}, cleaner.transform(df))
    assert p['rows_with_any_missing'] == 3
    assert p['total_missing_cells'] == 4


def test_timeout_checkpoint():
    # Exercise the actual supervisor with a deliberately stalled worker.
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        (folder/'supervisor.py').write_text((root/'supervisor.py').read_text())
        (folder/'main.py').write_text('''import sys,json,time
from pathlib import Path
out=Path(sys.argv[len(sys.argv)-1-sys.argv[::-1].index('--out-dir')+1]); out.mkdir(exist_ok=True)
c=out/'checkpoint_1'; c.mkdir()
(c/'predictions.csv').write_text('row_id,prediction\\n0,1\\n')
(c/'profile.json').write_text(json.dumps({'task':'classification','model_report':{'chosen_models':{'prior':1}}}))
(out/'checkpoint.json').write_text(json.dumps({'directory':'checkpoint_1'}))
time.sleep(60)
''')
        spec = importlib.util.spec_from_file_location('test_supervisor', folder/'supervisor.py')
        sup = importlib.util.module_from_spec(spec); spec.loader.exec_module(sup)
        started = time.monotonic()
        assert sup.run(['--out-dir',str(folder/'out'),'--time-budget','0.01']) == 0
        assert time.monotonic() - started < 12
        p = json.loads((folder/'out'/'profile.json').read_text())
        assert p['model_report']['supervisor_timeout'] is True
        assert (folder/'out'/'predictions.csv').read_text().endswith('0,1\n')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('PASS', name, flush=True)
