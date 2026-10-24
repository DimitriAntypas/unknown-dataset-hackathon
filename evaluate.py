"""Local benchmark: quality + runtime for every table in synth/ (run make_synthetic.py first)."""
import json, subprocess, sys, time
import numpy as np, pandas as pd
from sklearn.metrics import (accuracy_score, adjusted_rand_score, balanced_accuracy_score, f1_score, r2_score)

rows = []
failures = []
for m in json.load(open("synth/manifest.json")):
    n, task = m["name"], m["task"]
    t = time.time()
    if task == "clustering":
        args = ["--train", f"synth/{n}.csv", "--task", "clustering", "--n-clusters", str(m["k"]), "--id-col", "row_id"]
    else:
        args = ["--train", f"synth/{n}_train.csv", "--test", f"synth/{n}_test.csv", "--target", m.get("target", "target"),
                "--task", task]
        if n != "bike":
            args += ["--id-col", "row_id"]
    try:
        r = subprocess.run([sys.executable, "main.py", *args, "--out-dir", f"out/{n}"], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        failures.append(n)
        print(n, "FAILED: exceeded 60 seconds")
        continue
    wall = time.time() - t
    if r.returncode:
        failures.append(n); print(n, "FAILED\n", r.stderr[-1500:]); continue
    p = pd.read_csv(f"out/{n}/predictions.csv")
    if task == "clustering":
        y = pd.read_csv(f"synth/{n}_truth.csv")["y"]; rows.append((n, task, f"ARI={adjusted_rand_score(y, p['cluster']):.3f}", wall))
    else:
        y = pd.read_csv(f"synth/{n}_truth.csv")["target"]; pr = p["prediction"]
        if task == "classification":
            s = f"acc={accuracy_score(y, pr):.3f} bacc={balanced_accuracy_score(y, pr):.3f} f1m={f1_score(y, pr, average='macro'):.3f}"
        else:
            s = f"R2={r2_score(y, pr):.3f}"
        rows.append((n, task, s, wall))
for r in rows:
    print(f"{r[0]:8s} {r[1]:15s} {r[2]:40s} {r[3]:5.1f}s")
print("max wall:", round(max((r[3] for r in rows), default=0), 1), "s")

if failures:
    raise SystemExit("Failed datasets: " + ", ".join(failures))
