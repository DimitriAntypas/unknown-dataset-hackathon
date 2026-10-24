"""Regression coverage for the four defects found in the submission review.

Run with pytest, or directly with python tests/test_review_fixes.py.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import main as M


def test_sorted_integer_measurements_survive_cleaning():
    for values in (np.arange(100), np.arange(100)[::-1]):
        data = pd.DataFrame({"temperature": values, "id": np.arange(100)})
        cleaner = M.FeatureCleaner().fit(data)
        cleaned = cleaner.transform(data)
        assert "temperature" in cleaned, cleaner.dropped_
        np.testing.assert_array_equal(cleaned["temperature"], values)
        assert "id" not in cleaned


def test_single_column_csv_preserves_header_and_values():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "one_column.csv"
        expected = pd.DataFrame({"flag": ["True", "False", "?"]})
        expected.to_csv(path, index=False)
        pd.testing.assert_frame_equal(M.read_table(path), expected)


def test_single_column_csv_preserves_quoted_delimiters():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "quoted.csv"
        expected = pd.DataFrame({"feature": ["a,b", "c,d", "e,f"]})
        expected.to_csv(path, index=False)
        pd.testing.assert_frame_equal(M.read_table(path), expected)


def test_supported_delimiters_preserve_table():
    expected = pd.DataFrame({"feature": [1, 2, 3], "category": ["a", "b", "c"]})
    with tempfile.TemporaryDirectory() as tmp:
        for i, sep in enumerate((",", ";", "\t", "|")):
            path = Path(tmp) / f"table_{i}.csv"
            expected.to_csv(path, sep=sep, index=False)
            pd.testing.assert_frame_equal(M.read_table(path), expected)
        path = Path(tmp) / "table.tsv"
        expected.to_csv(path, sep="\t", index=False)
        pd.testing.assert_frame_equal(M.read_table(path), expected)


def test_boolean_transform_accepts_strings_and_missing_values():
    cleaner = M.FeatureCleaner().fit(pd.DataFrame({"flag": [True, False] * 20}))
    test = pd.DataFrame({"flag": ["True", " false ", "?", "", None,
                                  True, False, 1, 0, 1.0, 0.0, "YES", "no"]})
    expected = [1.0, 0.0, np.nan, np.nan, np.nan, 1.0, 0.0, 1.0, 0.0,
                1.0, 0.0, 1.0, 0.0]
    np.testing.assert_allclose(cleaner.transform(test)["flag"], expected, equal_nan=True)
    missing_column = cleaner.transform(pd.DataFrame({"other": [0, 1, 2]}))
    assert missing_column["flag"].isna().all()


def test_boolean_cli_with_missing_test_markers():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        train = pd.DataFrame({"flag": [True, False] * 100,
                              "noise": np.random.RandomState(2).normal(size=200),
                              "target": [1, 0] * 100})
        test = pd.DataFrame({"flag": ["True", "False", "?", ""],
                             "noise": [0.2, 0.5, -0.1, 0.0]})
        train.to_csv(folder / "train.csv", index=False)
        test.to_csv(folder / "test.csv", index=False)
        result = subprocess.run([
            sys.executable, str(ROOT / "main.py"), "--train", str(folder / "train.csv"),
            "--test", str(folder / "test.csv"), "--target", "target",
            "--task", "classification", "--time-budget", "5", "--out-dir", str(folder / "out"),
        ], capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stderr
        predictions = pd.read_csv(folder / "out" / "predictions.csv")
        assert len(predictions) == len(test)
        assert predictions.prediction.iloc[:2].tolist() == [1, 0]
        assert np.isfinite(predictions.filter(like="proba_").to_numpy()).all()


def test_sorted_predictor_cli_uses_signal_on_unseen_rows():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        train_values = np.arange(0, 400, 2)
        test_values = np.arange(1, 400, 2)
        pd.DataFrame({"temperature": train_values,
                      "target": (train_values >= 200).astype(int)}).to_csv(folder / "train.csv", index=False)
        pd.DataFrame({"temperature": test_values}).to_csv(folder / "test.csv", index=False)
        result = subprocess.run([
            sys.executable, str(ROOT / "main.py"), "--train", str(folder / "train.csv"),
            "--test", str(folder / "test.csv"), "--target", "target",
            "--task", "classification", "--time-budget", "5", "--out-dir", str(folder / "out"),
        ], capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stderr
        predictions = pd.read_csv(folder / "out" / "predictions.csv")
        assert len(predictions) == len(test_values)
        accuracy = np.mean(predictions.prediction.to_numpy() == (test_values >= 200))
        assert accuracy >= 0.95, accuracy
        profile = json.loads((folder / "out" / "profile.json").read_text())
        assert "temperature" not in profile["id_like_columns"]
        assert "prior" not in profile["model_report"]["chosen_models"]


def test_missing_profile_counts_datetime_markers():
    n = 100
    dates = pd.Series(pd.date_range("2020-01-01", periods=n).strftime("%Y-%m-%d"), dtype=object)
    dates.iloc[3] = "?"
    dates.iloc[4] = " "
    numbers = pd.Series([str(i) for i in range(n)], dtype=object)
    numbers.iloc[3] = "?"
    numbers.iloc[5] = None
    flags = pd.Series([True, False] * 50, dtype="boolean")
    flags.iloc[6] = pd.NA
    raw = pd.DataFrame({"when": dates, "number": numbers, "flag": flags})
    cleaner = M.FeatureCleaner().fit(raw)
    profile = M.build_profile(raw, cleaner, None, "clustering", 2, None, {}, cleaner.transform(raw))
    assert profile["column_types"]["when"] == "datetime"
    assert profile["missing_counts"] == {"when": 2, "number": 2, "flag": 1}
    assert profile["total_missing_cells"] == 5
    assert profile["rows_with_any_missing"] == 4
    assert profile["missing_fraction"] == {"when": 0.02, "number": 0.02, "flag": 0.01}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name, flush=True)
