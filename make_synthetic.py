"""Benchmark tables mimicking the hackathon's difficulties (scale, missing, constant/duplicate/ID/noise
columns, imbalance, non-linearity, datetimes). Writes synth/*.csv and synth/manifest.json."""
import json, os
import numpy as np, pandas as pd
from sklearn import datasets as D

rng = np.random.RandomState(0)
os.makedirs("synth", exist_ok=True)
manifest = []


def mess(df, noise=3, n_cat=2, miss=0.08, dup=True):
    df = df.copy()
    first = df.columns[0]
    df[first] = df[first] * 1000                      # wild scale
    df["const_col"] = 7
    if dup:
        df["dup_col"] = df[df.columns[1]] * 2.5 + 3   # scaled duplicate
    df["row_id"] = np.arange(len(df))
    for j in range(noise):
        df[f"noise{j}"] = rng.randn(len(df))
    for j in range(n_cat):
        df[f"cat{j}"] = rng.choice(list("abcde"), len(df))
    for c in list(df.columns[:3]) + ["cat0"]:
        df.loc[rng.rand(len(df)) < miss, c] = np.nan
    return df


def save_sup(name, task, X, y, test_frac=0.4, **kw):
    df = mess(pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]), **kw)
    df["target"] = y
    idx = rng.permutation(len(df)); cut = int(len(df) * (1 - test_frac))
    tr, te = df.iloc[idx[:cut]], df.iloc[idx[cut:]]
    tr.to_csv(f"synth/{name}_train.csv", index=False)
    te.drop(columns="target").to_csv(f"synth/{name}_test.csv", index=False)
    te[["target"]].to_csv(f"synth/{name}_truth.csv", index=False)
    manifest.append(dict(name=name, task=task))


def save_clu(name, X, y, k, **kw):
    df = mess(pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])]), **kw)
    df.to_csv(f"synth/{name}.csv", index=False)
    pd.DataFrame({"y": y}).to_csv(f"synth/{name}_truth.csv", index=False)
    manifest.append(dict(name=name, task="clustering", k=k))


# --- classification
d = D.load_breast_cancer(); save_sup("bc", "classification", d.data[:, :12], d.target)
d = D.load_wine(); save_sup("wine", "classification", d.data, d.target)
X, y = D.make_classification(3000, 14, n_informative=5, weights=[0.9, 0.1], flip_y=0.02, random_state=1)
save_sup("imb", "classification", X, y)
X, y = D.make_moons(1500, noise=0.25, random_state=1); X = np.hstack([X, rng.randn(1500, 4)])
save_sup("moons", "classification", X, y, noise=2)
# --- regression
d = D.load_diabetes(); save_sup("diab", "regression", d.data, d.target)
X, y = D.make_friedman1(2500, 10, noise=1.0, random_state=1); save_sup("fried", "regression", X, y)
X = rng.rand(3000, 4); dates = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.randint(0, 24 * 700, 3000), unit="h")
y = 50 + 30 * np.sin(2 * np.pi * dates.hour / 24) + 15 * (dates.dayofweek >= 5) + 40 * X[:, 0] + rng.randn(3000) * 4
df = pd.DataFrame(X, columns=list("abcd")); df.insert(0, "datetime", dates.strftime("%Y-%m-%d %H:%M:%S")); df["count"] = np.exp(y / 40)
idx = rng.permutation(3000); df.iloc[idx[:1800]].to_csv("synth/bike_train.csv", index=False)
df.iloc[idx[1800:]].drop(columns="count").to_csv("synth/bike_test.csv", index=False)
df.iloc[idx[1800:]][["count"]].rename(columns={"count": "target"}).to_csv("synth/bike_truth.csv", index=False)
manifest.append(dict(name="bike", task="regression", target="count"))
# --- clustering
d = D.load_iris(); save_clu("iris", d.data, d.target, 3, noise=2)
X, y = D.make_blobs(900, 5, centers=4, cluster_std=1.3, random_state=0); X = np.hstack([X, rng.randn(900, 3)])
save_clu("blobs", X, y, 4)
X, y = D.make_moons(800, noise=0.07, random_state=0); save_clu("moons_c", X, y, 2, noise=1, n_cat=0, miss=0.0)
X, y = D.make_blobs(1200, 6, centers=5, cluster_std=[1, 2, 1.5, 0.7, 1.2], random_state=3); save_clu("blobs5", X, y, 5, noise=4)
json.dump(manifest, open("synth/manifest.json", "w"))
