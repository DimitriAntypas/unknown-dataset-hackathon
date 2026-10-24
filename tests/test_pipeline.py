"""Robustness tests: python -m pytest -q"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main as M  # noqa: E402

rng = np.random.RandomState(0)


def _clf_frame(n=300):
    X = pd.DataFrame({"a": rng.randn(n), "b": rng.randn(n) * 1e4, "c": rng.choice(["x", "y", "z"], n)})
    y = ((X["a"] + (X["c"] == "x") * 1.5 + rng.randn(n) * 0.3) > 0.5).astype(int)
    return X.assign(target=np.where(y == 1, "yes", "no"))


def test_string_labels_and_missing_test_columns():
    df = _clf_frame()
    test = df.drop(columns=["target", "b"]).iloc[:50]       # column b absent in test
    r = M.solve(df, test, target="target", time_budget=25)
    assert set(r["labels"]) <= {"yes", "no"} and len(r["labels"]) == 50
    assert r["profile"]["task"] == "classification"


def test_numeric_text_with_blanks_and_datetimes():
    n = 400
    df = pd.DataFrame({
        "when": pd.date_range("2021-01-01", periods=n, freq="6h").strftime("%Y-%m-%d %H:%M"),
        "charges": [" " if i % 25 == 0 else f"{rng.rand() * 100:.2f}" for i in range(n)],
        "u": rng.randn(n)})
    df["target"] = 3 * pd.to_numeric(df["charges"], errors="coerce").fillna(50) + df["u"] * 5
    r = M.solve(df, df.drop(columns="target").iloc[:40], target="target", task="regression", time_budget=25)
    p = r["profile"]
    assert "charges" in p["numeric_stored_as_text"] and "when" in p["datetime_columns"]
    assert np.isfinite(r["labels"]).all()


def test_profile_flags_constant_duplicate_id_imbalance():
    n = 500
    base = rng.randn(n)
    df = pd.DataFrame({"id": np.arange(n), "x": base, "x_copy": base, "x_scaled": base * 3 + 1,
                       "const": 1, "noise": rng.randn(n)})
    df["target"] = (base + rng.randn(n) * 0.2 > 1.3).astype(int)
    p = M.solve(df, None, target="target", time_budget=25)["profile"]
    assert "const" in p["constant_columns"]
    assert ["x", "x_copy"] in p["duplicate_columns"]
    assert any(pair[0] == "x" and pair[1] == "x_scaled" for pair in p["perfectly_correlated_columns"])
    assert "id" in p["id_like_columns"]
    assert p["is_imbalanced"] is True


def test_tiny_dataset_and_single_class_do_not_crash():
    tiny = pd.DataFrame({"a": rng.randn(12), "target": [0, 1] * 6})
    assert len(M.solve(tiny, tiny.drop(columns="target"), target="target", time_budget=10)["labels"]) == 12
    one = pd.DataFrame({"a": rng.randn(40), "target": ["k"] * 40})
    r = M.solve(one, one.drop(columns="target"), target="target", time_budget=10)
    assert set(r["labels"]) == {"k"}


def test_high_cardinality_and_all_missing_column():
    n = 300
    df = pd.DataFrame({"city": [f"c{i % 60}" for i in range(n)], "empty": np.nan, "v": rng.randn(n)})
    df["target"] = df["v"] * 2 + (df["city"].str[1:].astype(int) % 7)
    r = M.solve(df, df.drop(columns="target"), target="target", task="regression", time_budget=25)
    assert "empty" in r["profile"]["constant_columns"] and "city" in r["profile"]["high_cardinality_columns"]


def test_clustering_returns_exactly_k_labels_and_handles_missing():
    from sklearn.datasets import make_blobs
    X, y = make_blobs(300, 3, centers=3, random_state=1)
    df = pd.DataFrame(X, columns=list("abc"))
    df.loc[::17, "a"] = np.nan
    df["cat"] = rng.choice(["p", "q"], len(df))
    r = M.solve(df, None, task="clustering", n_clusters=3, time_budget=25)
    from sklearn.metrics import adjusted_rand_score
    assert len(set(r["labels"])) == 3 and adjusted_rand_score(y, r["labels"]) > 0.9


def test_cli_end_to_end(tmp_path):
    df = _clf_frame(200)
    tr, te = tmp_path / "tr.csv", tmp_path / "te.csv"
    df.to_csv(tr, index=False)
    df.drop(columns="target").to_csv(te, index=False)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, os.path.join(root, "main.py"), "--train", str(tr), "--test", str(te),
                        "--target", "target", "--out-dir", str(tmp_path / "o")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert len(pd.read_csv(tmp_path / "o" / "predictions.csv")) == 200
    assert "model_report" in json.load(open(tmp_path / "o" / "profile.json"))


if __name__ == "__main__":  # works without pytest: python tests/test_pipeline.py
    import inspect
    import pathlib
    import tempfile
    import traceback
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                kw = {"tmp_path": pathlib.Path(tempfile.mkdtemp())} if "tmp_path" in inspect.signature(fn).parameters else {}
                fn(**kw)
                print("PASS", name)
            except Exception:
                failed += 1
                print("FAIL", name)
                traceback.print_exc()
    sys.exit(1 if failed else 0)
