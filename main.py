#!/usr/bin/env python3
"""Unknown Dataset Hackathon - automatic machine learning on unfamiliar tables.

Supervised:  python main.py --train train.csv --test test.csv --target y --out-dir out
Clustering:  python main.py --train data.csv --task clustering --n-clusters 4 --out-dir out

Writes  <out-dir>/predictions.csv  and  <out-dir>/profile.json.
Adapt CLI arguments, metadata parsing, profile fields and I/O if the official contract differs.
"""
# Start the CLI supervisor before importing numerical libraries.
if __name__ == "__main__":
    import sys
    if "--_worker" not in sys.argv:
        from supervisor import run
        raise SystemExit(run(sys.argv[1:]))
    sys.argv.remove("--_worker")

import argparse
import csv
import json
import os
import time
import warnings

# Respect an explicitly configured thread limit; otherwise use one CPU thread.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from scipy.stats import skew as _skew
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (accuracy_score, adjusted_rand_score, balanced_accuracy_score,
                             f1_score, r2_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, OneHotEncoder, PolynomialFeatures,
                                   StandardScaler)

SEED = 42
START = time.monotonic()  # legacy import timestamp; budgets are per invocation
MISSING_TOKEN = "__missing__"


# =============================================================================== time budget
class Budget:
    """Wall-clock guard. The contest limit is 60 s per dataset; we plan for `total` (default 40 s)."""

    def __init__(self, total=40.0):
        self.total = float(total)
        if not np.isfinite(self.total) or self.total <= 0:
            raise ValueError("time_budget must be finite and positive")
        self.started = time.monotonic()

    def elapsed(self):
        return time.monotonic() - self.started

    def left(self):
        return self.total - self.elapsed()

    def select_open(self):  # model search stops after ~half the budget
        return self.elapsed() < 0.5 * self.total

    def select_hard_stop(self):  # abort a candidate that is still running after ~70 %
        return self.elapsed() > 0.7 * self.total


# =============================================================================== feature cleaning
class FeatureCleaner:
    """Learns column types and a cleaning plan on training features; applies it consistently to any table.

    Handles: numbers stored as text, datetime strings, booleans, blank/`?` missing markers, infinities,
    constant / duplicate / perfectly-correlated / ID-like columns, high-cardinality categoricals (frequency
    encoding), and test tables with missing or extra columns.
    """
    DT_PARTS = ("year", "month", "day", "dayofweek", "hour", "epoch_days")

    # ---- type detection
    @staticmethod
    def _detect(s):
        if pd.api.types.is_bool_dtype(s):
            return "bool"
        if pd.api.types.is_datetime64_any_dtype(s):
            return "datetime"
        if pd.api.types.is_numeric_dtype(s):
            return "numeric"
        ss = s.dropna().astype(str).str.strip()
        ss = ss[ss != ""]
        if len(ss) == 0:
            return "categorical"
        if pd.to_numeric(ss, errors="coerce").notna().mean() > 0.95:
            return "numeric_text"
        sample = ss.sample(min(len(ss), 200), random_state=SEED)
        if sample.str.contains(r"\d").mean() > 0.9 and sample.str.len().median() >= 6:
            try:
                if pd.to_datetime(sample, errors="coerce").notna().mean() > 0.95:
                    return "datetime"
            except Exception:
                pass
        return "categorical"

    def _basic(self, X, fit):
        out = OrderedDict()
        for c in self.columns_:
            s = X[c] if c in X.columns else pd.Series(np.nan, index=X.index)
            kind = self.kinds_[c]
            if kind == "datetime":
                dt = pd.to_datetime(s, errors="coerce")
                try:
                    dt = dt.dt.tz_localize(None)
                except Exception:
                    pass
                parts = {"year": dt.dt.year, "month": dt.dt.month, "day": dt.dt.day,
                         "dayofweek": dt.dt.dayofweek, "hour": dt.dt.hour,
                         "epoch_days": (dt - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400.0}
                if fit:
                    self.dt_keep_[c] = [p for p in self.DT_PARTS if parts[p].nunique(dropna=True) > 1]
                for p in self.dt_keep_[c]:
                    out[f"{c}__{p}"] = parts[p].astype(float)
            elif kind == "bool":
                # CSV readers may return text when a test column has missing
                # markers even though its training counterpart is boolean.
                values = s.astype("string").str.strip().str.casefold()
                out[c] = values.map({
                    "true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0,
                    "1.0": 1.0, "0.0": 0.0, "yes": 1.0, "no": 0.0,
                }).astype(float)
            elif kind in ("numeric", "numeric_text"):
                out[c] = pd.to_numeric(s, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
            else:
                t = s.astype("string").str.strip()
                t = t.mask(t.isin(["", "?"]))
                out[c] = t.astype(object).where(t.notna(), np.nan)
        return pd.DataFrame(out, index=X.index)

    # ---- fit / transform
    def fit(self, X):
        self.columns_ = list(X.columns)
        self.kinds_ = {c: self._detect(X[c]) for c in X.columns}
        self.dt_keep_ = {}
        basic = self._basic(X, fit=True)
        n = len(basic)
        drops = OrderedDict()
        self.dup_pairs_, self.near_pairs_ = [], []
        is_num = {c: pd.api.types.is_numeric_dtype(basic[c]) for c in basic.columns}

        for c in basic.columns:  # constant / empty
            if basic[c].nunique(dropna=True) <= 1:
                drops[c] = "constant"
        for c in basic.columns:  # ID-like
            if c in drops:
                continue
            s = basic[c]
            nun = s.nunique(dropna=True)
            if not is_num[c]:
                if n > 20 and nun > 0.9 * n:
                    drops[c] = "id_like"
            else:
                raw_name = c.split("__")[0]
                name_idish = raw_name.lower() in {"id", "index", "idx", "rowid", "row_id", "unnamed: 0", "customerid"}
                intlike = s.notna().all() and np.allclose(s, np.round(s))
                # A sorted integer measurement can carry the entire predictive
                # signal. Require an ID-like name instead of relying on order.
                if intlike and nun == n and n > 20 and name_idish:
                    drops[c] = "id_like"
        seen = {}
        for c in basic.columns:  # exact duplicates
            if c in drops:
                continue
            s = basic[c]
            key = (is_num[c], int(pd.util.hash_pandas_object(s.astype(str), index=False).sum() % (2 ** 61)))
            if key in seen and s.astype(str).equals(basic[seen[key]].astype(str)):
                drops[c] = f"duplicate_of:{seen[key]}"
                self.dup_pairs_.append([seen[key], c])
            else:
                seen.setdefault(key, c)
        num_left = [c for c in basic.columns if c not in drops and is_num[c]]
        if 2 <= len(num_left) <= 400:  # perfectly correlated (scaled / shifted copies)
            M = basic[num_left].to_numpy(float)
            med = np.nanmedian(M, axis=0)
            M = np.where(np.isnan(M), med, M)
            sd = M.std(0)
            ok = sd > 1e-12
            Z = (M - M.mean(0)) / np.where(ok, sd, 1)
            C = np.abs(Z.T @ Z / n)
            for j in range(len(num_left)):
                if num_left[j] in drops:
                    continue
                for i in range(j):
                    if num_left[i] not in drops and ok[i] and ok[j] and C[i, j] > 0.9999:
                        drops[num_left[j]] = f"perfectly_correlated_with:{num_left[i]}"
                        self.near_pairs_.append([num_left[i], num_left[j]])
                        break
        self.dropped_ = drops
        self.out_columns_ = [c for c in basic.columns if c not in drops]
        self.freq_maps_ = {}
        for c in self.out_columns_:
            if not is_num[c] and basic[c].nunique() > 20:
                self.freq_maps_[c] = basic[c].value_counts(normalize=True).to_dict()
        self.basic_ = basic  # kept for profiling; released by the caller
        self.is_num_ = is_num
        return self

    def transform(self, X):
        basic = self._basic(X, fit=False)
        out = basic[self.out_columns_].copy()
        for c, m in self.freq_maps_.items():
            out[f"{c}__freq"] = basic[c].map(m).fillna(0.0).astype(float)
        return out


# =============================================================================== model preprocessing
class SkewFix(BaseEstimator, TransformerMixin):
    """Signed log1p on strongly skewed columns (learned on training data)."""

    def __init__(self, thr=1.5):
        self.thr = thr

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        with np.errstate(all="ignore"):
            sk = _skew(X, axis=0, nan_policy="omit")
        self.cols_ = np.where(np.nan_to_num(np.abs(np.asarray(sk))) > self.thr)[0]
        return self

    def transform(self, X):
        X = np.array(X, float, copy=True)
        if len(self.cols_):
            X[:, self.cols_] = np.sign(X[:, self.cols_]) * np.log1p(np.abs(X[:, self.cols_]))
        return X


def _split_cols(X):
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    return num, [c for c in X.columns if c not in num]


def make_prep(X, family):
    """family: 'hgb' (native NaN), 'tree' (impute) or 'linear' (impute + skew fix + scale)."""
    num, cat = _split_cols(X)
    has_missing = bool(X[num].isna().any().any()) if num else False
    parts = []
    if num:
        if family == "hgb":
            parts.append(("num", "passthrough", num))
        elif family == "tree":
            parts.append(("num", SimpleImputer(strategy="median", add_indicator=has_missing,
                                               keep_empty_features=True), num))
        else:
            parts.append(("num", Pipeline([
                ("imp", SimpleImputer(strategy="median", add_indicator=has_missing, keep_empty_features=True)),
                ("skew", SkewFix()), ("sc", StandardScaler())]), num))
    if cat:
        parts.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value=MISSING_TOKEN)),
            ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, max_categories=20,
                                  sparse_output=False, drop="if_binary"))]), cat))
    return ColumnTransformer(parts, remainder="drop", sparse_threshold=0.0, verbose_feature_names_out=False)


def est_dim(X):
    num, cat = _split_cols(X)
    return len(num) + sum(min(X[c].nunique(), 20) for c in cat)


# =============================================================================== supervised candidates
def build_candidates(task, X, y, n_classes=None):
    n = len(X)
    dim = est_dim(X)
    es = n >= 600
    cands = OrderedDict()
    P = lambda fam, m: Pipeline([("prep", make_prep(X, fam)), ("m", m)])
    if task == "classification":
        cands["hgb"] = P("hgb", HistGradientBoostingClassifier(
            learning_rate=0.06, max_iter=300, l2_regularization=1.0, early_stopping=es,
            n_iter_no_change=20, random_state=SEED))
        cands["logreg"] = P("linear", LogisticRegression(C=1.0, max_iter=500))
        cands["hgb_small"] = P("hgb", HistGradientBoostingClassifier(
            learning_rate=0.1, max_iter=200, max_leaf_nodes=8, min_samples_leaf=10, l2_regularization=2.0,
            early_stopping=es, n_iter_no_change=15, random_state=SEED))
        cands["rf"] = P("tree", RandomForestClassifier(n_estimators=120, min_samples_leaf=1, n_jobs=1,
                                                       random_state=SEED))
        cands["et"] = P("tree", ExtraTreesClassifier(n_estimators=120, min_samples_leaf=1, n_jobs=1,
                                                     random_state=SEED))
        if n <= 30000:
            cands["mlp"] = P("linear", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                                                     max_iter=200, random_state=SEED))
        if dim <= 40 and n <= 30000:
            cands["knn"] = P("linear", KNeighborsClassifier(n_neighbors=15, weights="distance"))
        if dim <= 22 and n <= 8000:
            cands["logreg_poly"] = Pipeline([("prep", make_prep(X, "linear")),
                                             ("poly", PolynomialFeatures(2, include_bias=False)),
                                             ("sc", StandardScaler()),
                                             ("m", LogisticRegression(C=0.2, max_iter=500))])
    else:
        cands["hgb"] = P("hgb", HistGradientBoostingRegressor(
            learning_rate=0.06, max_iter=400, l2_regularization=1.0, early_stopping=es,
            n_iter_no_change=20, random_state=SEED))
        cands["ridge"] = P("linear", RidgeCV(alphas=np.logspace(-3, 3, 13)))
        cands["hgb_small"] = P("hgb", HistGradientBoostingRegressor(
            learning_rate=0.1, max_iter=250, max_leaf_nodes=8, min_samples_leaf=10, l2_regularization=2.0,
            early_stopping=es, n_iter_no_change=15, random_state=SEED))
        if np.nanmin(y) >= 0 and abs(float(pd.Series(y).skew())) > 1.0:
            cands["hgb_log"] = TransformedTargetRegressor(
                regressor=cands["hgb"], func=np.log1p, inverse_func=np.expm1, check_inverse=False)
        cands["rf"] = P("tree", RandomForestRegressor(n_estimators=120, min_samples_leaf=1, n_jobs=1,
                                                      random_state=SEED))
        cands["et"] = P("tree", ExtraTreesRegressor(n_estimators=120, min_samples_leaf=1, n_jobs=1,
                                                    random_state=SEED))
        if n <= 30000:
            cands["mlp"] = TransformedTargetRegressor(
                regressor=P("linear", MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True,
                                                   max_iter=300, random_state=SEED)),
                transformer=StandardScaler())
        if dim <= 40 and n <= 30000:
            cands["knn"] = P("linear", KNeighborsRegressor(n_neighbors=10, weights="distance"))
        if dim <= 30 and n <= 8000:
            cands["ridge_poly"] = Pipeline([("prep", make_prep(X, "linear")),
                                            ("poly", PolynomialFeatures(2, include_bias=False)),
                                            ("sc", StandardScaler()),
                                            ("m", RidgeCV(alphas=np.logspace(-1, 4, 11)))])
    return cands


def _logloss(P, y):
    P = np.clip(P, 1e-15, 1.0)
    P = P / P.sum(1, keepdims=True)
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))


def _rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def _fold_iter(task, y, k):
    if task == "classification":
        mc = np.bincount(y).min()
        if mc >= 2:
            return list(StratifiedKFold(int(min(k, mc)), shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    return list(KFold(k, shuffle=True, random_state=SEED).split(np.zeros(len(y))))


def _classes_of(model):
    return model.classes_ if hasattr(model, "classes_") else model[-1].classes_


def _predict_clf(model, Xv, n_classes):
    p = model.predict_proba(Xv)
    full = np.zeros((len(Xv), n_classes))
    full[:, np.asarray(_classes_of(model), int)] = p
    return full


def greedy_blend(oofs, loss_fn, steps=20):
    chosen, cur, best_loss = [], None, np.inf
    for _ in range(steps):
        best = None
        for nm, o in oofs.items():
            cand = o if cur is None else (cur * len(chosen) + o) / (len(chosen) + 1)
            l = loss_fn(cand)
            if best is None or l < best[0]:
                best = (l, nm, cand)
        if chosen and best[0] >= best_loss - 1e-9:
            break
        chosen.append(best[1])
        cur, best_loss = best[2], best[0]
    cnt = Counter(chosen)
    tot = sum(cnt.values())
    return {k: v / tot for k, v in cnt.items()}, best_loss


def _decision_score(y, pred):
    return float(np.mean([accuracy_score(y, pred), balanced_accuracy_score(y, pred),
                          f1_score(y, pred, average="macro")]))


def tune_class_weights(P, y):
    """Prior-shift weights for imbalanced data, tuned on out-of-fold probabilities."""
    K = P.shape[1]
    w = np.ones(K)
    base = _decision_score(y, (P * w).argmax(1))
    best = base
    grid = [0.25, 0.4, 0.6, 0.8, 1.25, 1.6, 2.5, 4.0]
    for _ in range(2):
        for k in range(K):
            for g in grid:
                w2 = w.copy()
                w2[k] *= g
                sc = _decision_score(y, (P * w2).argmax(1))
                if sc > best + 1e-6:
                    best, w = sc, w2
    if best - base < 0.004:  # not a real gain -> keep plain argmax
        return np.ones(K), base, base
    return w / w.max(), base, best


def run_supervised(Xtr, ytr_raw, Xte, task, budget, info):
    n_full = len(Xtr)
    if task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(ytr_raw)
        K = len(le.classes_)
    else:
        y = np.asarray(ytr_raw, float)
        le, K = None, None

    if n_full < 8 or (task == "classification" and K < 2) or Xtr.shape[1] == 0:
        return _prior_prediction(task, y, le, K, len(Xte), info)

    # selection subsample
    sel_n = 6000
    if n_full > sel_n:
        rng = np.random.RandomState(SEED)
        if task == "classification":
            from sklearn.model_selection import train_test_split
            idx, _ = train_test_split(np.arange(n_full), train_size=sel_n, stratify=y, random_state=SEED) \
                if np.bincount(y).min() >= 2 else (rng.choice(n_full, sel_n, replace=False), None)
        else:
            idx = rng.choice(n_full, sel_n, replace=False)
        idx = np.sort(idx)
    else:
        idx = np.arange(n_full)
    Xs, ys = Xtr.iloc[idx].reset_index(drop=True), y[idx]
    n_sel = len(idx)
    k = 5 if n_sel <= 3000 else 3
    folds = _fold_iter(task, ys, k)

    cands = build_candidates(task, Xs, ys, K)
    loss_fn = (lambda P: _logloss(P, ys)) if task == "classification" else (lambda p: _rmse(p, ys))
    oofs, cv_time, errors = {}, {}, {}
    for name, est in cands.items():
        if oofs and not budget.select_open():
            break
        oof = np.zeros((n_sel, K)) if task == "classification" else np.zeros(n_sel)
        t0 = time.time()
        try:
            aborted = False
            for tr, va in folds:
                m = clone(est).fit(Xs.iloc[tr], ys[tr])
                if task == "classification":
                    oof[va] = _predict_clf(m, Xs.iloc[va], K)
                else:
                    oof[va] = np.asarray(m.predict(Xs.iloc[va]), float)
                if oofs and budget.select_hard_stop():
                    aborted = True
                    break
            if aborted or not np.all(np.isfinite(oof)):
                continue
            oofs[name] = oof
            cv_time[name] = time.time() - t0
        except Exception as e:
            errors[name] = str(e)[:160]
    if errors:
        info["model_errors"] = errors
    if not oofs:
        return _prior_prediction(task, y, le, K, len(Xte), info)

    scores = {nm: loss_fn(o) for nm, o in oofs.items()}
    weights, blend_loss = greedy_blend(oofs, loss_fn)
    info["cv_metric"] = "log_loss" if task == "classification" else "rmse"
    info["cv_scores"] = {k_: round(v, 5) for k_, v in sorted(scores.items(), key=lambda kv: kv[1])}
    info["blend_weights"] = {k_: round(v, 3) for k_, v in weights.items()}
    info["blend_cv_score"] = round(blend_loss, 5)
    info["n_rows_used_for_selection"] = int(n_sel)
    info["cv_seconds"] = {k_: round(v, 2) for k_, v in cv_time.items()}

    blend_oof = sum(weights[nm] * oofs[nm] for nm in weights)
    if task == "classification":
        info["cv_accuracy"] = round(float(accuracy_score(ys, blend_oof.argmax(1))), 4)
        info["cv_balanced_accuracy"] = round(float(balanced_accuracy_score(ys, blend_oof.argmax(1))), 4)
        cw = np.ones(K)
        share = np.bincount(ys, minlength=K) / n_sel
        if share.max() / max(share.min(), 1e-9) > 2.5 and np.bincount(ys).min() >= 5:
            cw, base, best = tune_class_weights(blend_oof, ys)
            info["class_weight_tuning"] = {"weights": [round(float(x), 3) for x in cw],
                                           "score_before": round(base, 4), "score_after": round(best, 4)}
    else:
        info["cv_r2"] = round(float(r2_score(ys, blend_oof)), 4)
    # linear vs non-linear signal (profile hint)
    lin = [scores[m] for m in scores if m in ("logreg", "ridge")]
    non = [scores[m] for m in scores if m in ("hgb", "hgb_small", "rf", "et", "mlp", "knn", "hgb_log")]
    if lin and non:
        info["nonlinear_signal"] = bool(min(non) < (0.95 if task == "classification" else 0.9) * min(lin))

    # final fits on all rows, respecting the clock
    order = sorted(weights, key=lambda m: -weights[m])
    preds, used = {}, {}
    scale = n_full / n_sel
    for nm in order:
        est_t = cv_time[nm] / len(folds) * scale * 1.3
        if preds and est_t > budget.left() - 4:
            continue
        try:
            m = clone(cands[nm]).fit(Xtr, y)
            preds[nm] = _predict_clf(m, Xte, K) if task == "classification" else np.asarray(m.predict(Xte), float)
            used[nm] = weights[nm]
        except Exception as e:
            info.setdefault("model_errors", {})[nm + "_final"] = str(e)[:160]
    if not preds:  # last resort: fastest scored model
        nm = min(cv_time, key=cv_time.get)
        m = clone(cands[nm]).fit(Xtr, y)
        preds[nm] = _predict_clf(m, Xte, K) if task == "classification" else np.asarray(m.predict(Xte), float)
        used[nm] = 1.0
    tot = sum(used.values())
    final = sum(used[nm] / tot * preds[nm] for nm in preds)
    info["chosen_models"] = {k_: round(v / tot, 3) for k_, v in used.items()}
    if task == "classification":
        w = np.array(info.get("class_weight_tuning", {}).get("weights", np.ones(K)))
        out = le.inverse_transform((final * w).argmax(1))
        return out, final, le.classes_
    if np.nanmin(y) >= 0:
        final = np.clip(final, 0, None)
    return final, None, None


def _prior_prediction(task, y, le, K, n_out, info):
    info["note"] = "fallback to prior (too little data or no usable features)"
    info["chosen_models"] = {"prior": 1.0}
    if task == "classification":
        c = np.bincount(y, minlength=K) / len(y)
        proba = np.tile(c, (n_out, 1))
        return np.repeat(le.inverse_transform([int(np.argmax(c))]), n_out), proba, le.classes_
    return np.full(n_out, float(np.mean(y)) if len(y) else 0.0), None, None


# =============================================================================== clustering
def _structure_gain(Z, k):
    """BIC gain of a k-component 1-D mixture over one Gaussian; positive => feature has cluster structure."""
    gains = []
    for j in range(Z.shape[1]):
        v = Z[:, [j]]
        try:
            gains.append(GaussianMixture(1, random_state=SEED).fit(v).bic(v) -
                         GaussianMixture(min(k, 4), random_state=SEED, reg_covar=1e-4).fit(v).bic(v))
        except Exception:
            gains.append(-1e9)
    return np.array(gains)


def cluster_spaces(X, k):
    num, cat = _split_cols(X)
    spaces, meta = OrderedDict(), {}

    def enc(cols, scale_fam="linear"):
        sub = X[cols]
        P = make_prep(sub, scale_fam)
        Z = np.nan_to_num(P.fit_transform(sub))
        return Z[:, Z.std(0) > 1e-8]

    if num:
        sub = X[num]
        if sub.isna().any().any() and len(num) <= 60 and len(sub) <= 20000 and len(num) > 1:
            imp = IterativeImputer(max_iter=8, random_state=SEED, skip_complete=True)
        else:
            imp = SimpleImputer(strategy="median", keep_empty_features=True)
        Zn = np.nan_to_num(Pipeline([("imp", imp), ("skew", SkewFix()), ("sc", StandardScaler())]).fit_transform(sub))
        Zn = Zn[:, Zn.std(0) > 1e-8]
        names = [c for c, sd in zip(num, np.nan_to_num(Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True))]).fit_transform(sub)).std(0)) if sd > 1e-8]
        if Zn.shape[1]:
            spaces["num"] = Zn
            if Zn.shape[1] > 3:
                g = _structure_gain(Zn, k)
                keep = np.argsort(-g)[:max(2, int(np.ceil(Zn.shape[1] / 2)))]
                keep = [j for j in keep if g[j] > 0] or list(range(Zn.shape[1]))
                spaces["num_sel"] = Zn[:, sorted(keep)]
                meta["unstructured_features"] = [names[j] for j in range(min(len(names), len(g))) if g[j] <= 0]
    if cat:
        Za = enc(list(X.columns))
        if Za.shape[1]:
            spaces["all"] = Za
            if Za.shape[1] > 6:
                spaces["all_pca"] = PCA(n_components=0.9, random_state=SEED).fit_transform(Za)
    return spaces, meta


def ensure_k(labels, k):
    """Split the largest groups deterministically when identical points collapse clusters."""
    labels = LabelEncoder().fit_transform(labels)
    if not 1 <= k <= len(labels):
        raise ValueError("n_clusters must be between 1 and the number of rows")
    while len(np.unique(labels)) < k:
        sizes = np.bincount(labels)
        members = np.flatnonzero(labels == sizes.argmax())
        labels[members[-1]] = len(sizes)
    return labels


def run_clustering(X, k, budget, info):
    n = len(X)
    if not 1 <= k <= n:
        raise ValueError("n_clusters must be between 1 and the number of rows")
    if k == 1 or k == n:
        info["note"] = "No varying features; deterministic partition has no learned structure"
        return np.arange(n) % k if k == 1 else np.arange(n)
    spaces, meta = cluster_spaces(X, k)
    info.update(meta)
    if not spaces:
        info["note"] = "No varying features; deterministic partition has no learned structure"
        return np.arange(n) % k
    rng = np.random.RandomState(SEED)
    sil_idx = rng.choice(n, min(n, 2500), replace=False)
    cands = []  # (name, labels, silhouette, purity)
    for sname, Z in spaces.items():
        d = Z.shape[1]
        nn_idx = NearestNeighbors(n_neighbors=min(11, max(2, len(sil_idx) - 1))).fit(Z[sil_idx]).kneighbors(
            Z[sil_idx], return_distance=False)[:, 1:]
        algos = OrderedDict()
        algos["kmeans"] = lambda: KMeans(k, n_init=10, random_state=SEED)
        for cov in ("spherical", "diag", "full"):
            if cov == "full" and d > 15:
                continue
            algos[f"gmm_{cov}"] = (lambda c=cov: GaussianMixture(k, covariance_type=c, n_init=2, random_state=SEED,
                                                                 reg_covar=1e-4))
        if n <= 5000:
            algos["ward"] = lambda: AgglomerativeClustering(k, linkage="ward")
        if n <= 2000 and sname in ("num", "num_sel", "all") and d >= 2:
            algos["spectral"] = lambda: SpectralClustering(
                k, affinity="nearest_neighbors", n_neighbors=min(15, max(5, n // 50)), random_state=SEED)
        for aname, mk in algos.items():
            if cands and budget.elapsed() > 0.6 * budget.total:
                break
            try:
                lab = mk().fit_predict(Z)
                if len(np.unique(lab)) < 2:
                    continue
                s = float(silhouette_score(Z[sil_idx], lab[sil_idx])) if len(np.unique(lab[sil_idx])) > 1 else -1
                ls = np.asarray(lab)[sil_idx]
                purity = float(np.mean(ls[nn_idx] == ls[:, None]))
                cands.append((f"{aname}/{sname}", np.asarray(lab), s, purity))
            except Exception as e:
                info.setdefault("model_errors", {})[f"{aname}/{sname}"] = str(e)[:160]
    if not cands:
        return ensure_k(KMeans(k, n_init=5, random_state=SEED).fit_predict(next(iter(spaces.values()))), k)
    # consensus: how well does each labelling agree with all the others
    m = len(cands)
    agree = np.zeros(m)
    for i in range(m):
        if m > 1:
            agree[i] = np.mean([adjusted_rand_score(cands[i][1], cands[j][1]) for j in range(m) if j != i])
    scored = []
    for i, (name, lab, s, pur) in enumerate(cands):
        sizes = np.bincount(lab)
        penalty = 0.0
        if len(np.unique(lab)) != k:
            penalty += 0.3
        if sizes[sizes > 0].min() < 0.01 * n:
            penalty += 0.15
        scored.append((0.4 * s + 0.3 * agree[i] + 0.3 * pur - penalty, name, lab, s, agree[i], pur))
    scored.sort(key=lambda t: -t[0])
    best = scored[0]
    info["cluster_candidates"] = {nm: {"score": round(sc, 4), "silhouette": round(s, 4), "consensus_ari": round(a, 4),
                                       "neighbour_purity": round(p, 4)} for sc, nm, _, s, a, p in scored[:12]}
    info["chosen_models"] = {best[1]: 1.0}
    info["silhouette"] = round(best[3], 4)
    return ensure_k(best[2], k)


# =============================================================================== profiling
def _f(x):
    try:
        x = float(x)
    except Exception:
        return None
    return None if np.isnan(x) or np.isinf(x) else round(x, 6)


def build_profile(raw_X, cleaner, y, task, k, target, info, Xclean, relevance=True):
    basic = cleaner.basic_
    n = len(raw_X)
    num_cols = [c for c in basic.columns if cleaner.is_num_[c]]
    cat_cols = [c for c in basic.columns if not cleaner.is_num_[c]]
    raw_kind = cleaner.kinds_
    prof = OrderedDict()
    prof["task"] = task
    prof["n_rows"] = int(n)
    prof["n_columns"] = int(raw_X.shape[1])
    if target is not None:
        prof["target_column"] = target
    if k:
        prof["n_clusters_requested"] = int(k)
    prof["column_types"] = {c: raw_kind[c] for c in raw_X.columns}
    prof["numeric_columns"] = [c for c in raw_X.columns if raw_kind[c] in ("numeric", "numeric_text")]
    prof["categorical_columns"] = [c for c in raw_X.columns if raw_kind[c] == "categorical"]
    prof["boolean_columns"] = [c for c in raw_X.columns if raw_kind[c] == "bool"]
    prof["datetime_columns"] = [c for c in raw_X.columns if raw_kind[c] == "datetime"]
    prof["numeric_stored_as_text"] = [c for c in raw_X.columns if raw_kind[c] == "numeric_text"]
    # One normalized mask supplies every missing-value statistic, including
    # markers in datetime columns and nullable boolean values.
    missing_mask = raw_X.isna().copy()
    for c in raw_X.columns:
        missing_mask[c] |= raw_X[c].astype("string").str.strip().isin(["", "?"])
    miss = {c: int(missing_mask[c].sum()) for c in raw_X.columns}
    prof["missing_counts"] = {c: v for c, v in miss.items() if v > 0}
    prof["missing_fraction"] = {c: round(v / n, 4) for c, v in miss.items() if v > 0}
    prof["columns_with_missing"] = list(prof["missing_counts"])
    prof["total_missing_cells"] = int(sum(miss.values()))
    prof["rows_with_any_missing"] = int(missing_mask.any(axis=1).sum())
    prof["constant_columns"] = [c for c, r in cleaner.dropped_.items() if r == "constant"]
    prof["duplicate_columns"] = cleaner.dup_pairs_
    prof["perfectly_correlated_columns"] = cleaner.near_pairs_
    prof["id_like_columns"] = [c for c, r in cleaner.dropped_.items() if r == "id_like"]
    prof["n_duplicate_rows"] = int(raw_X.duplicated().sum())
    prof["dropped_columns"] = dict(cleaner.dropped_)
    prof["high_cardinality_columns"] = [c for c in cat_cols if basic[c].nunique() > 20]
    prof["cardinality"] = {c: int(basic[c].nunique()) for c in cat_cols}
    if num_cols:
        B = basic[num_cols]
        sd = B.std()
        pos = sd[sd > 1e-12]
        prof["numeric_summary"] = {c: {"mean": _f(B[c].mean()), "std": _f(sd[c]), "min": _f(B[c].min()),
                                       "max": _f(B[c].max()), "skew": _f(B[c].skew())} for c in num_cols}
        prof["scale_ratio_max_over_min_std"] = _f(pos.max() / pos.min()) if len(pos) > 1 else 1.0
        prof["differently_scaled_features"] = bool(len(pos) > 1 and pos.max() / pos.min() > 20)
        prof["skewed_columns"] = [c for c in num_cols if abs(_f(B[c].skew()) or 0) > 1.0]
        q1, q3 = B.quantile(0.25), B.quantile(0.75)
        iqr = q3 - q1
        out_ct = ((B.lt(q1 - 3 * iqr)) | (B.gt(q3 + 3 * iqr))).sum()
        prof["outlier_counts_3iqr"] = {c: int(v) for c, v in out_ct.items() if v > 0 and iqr[c] > 0}
    prof["n_features_after_cleaning"] = int(Xclean.shape[1])
    if y is not None:
        if task == "classification":
            vc = pd.Series(y).value_counts(normalize=True)
            cnt = pd.Series(y).value_counts()
            prof["n_classes"] = int(len(vc))
            prof["class_counts"] = {str(a): int(b) for a, b in cnt.items()}
            prof["class_distribution"] = {str(a): round(float(b), 4) for a, b in vc.items()}
            prof["imbalance_ratio"] = _f(vc.max() / vc.min())
            prof["minority_class"] = str(vc.idxmin())
            prof["is_imbalanced"] = bool(vc.max() / vc.min() > 3)
        else:
            ys = pd.Series(y, dtype=float)
            prof["target_summary"] = {s: _f(getattr(ys, s)()) for s in ("mean", "std", "min", "max", "skew")}
            prof["target_skewed"] = bool(abs(_f(ys.skew()) or 0) > 1.0)
        prof["likely_irrelevant_features"] = _irrelevant(Xclean, y, task) if relevance else []
        prof["irrelevance_analysis_completed"] = bool(relevance)
    prof["model_report"] = info
    return prof


def _irrelevant(X, y, task, budget_rows=1500, max_cols=150):
    """Features whose mutual information with the target is no better than with a shuffled target."""
    try:
        if X.shape[1] == 0 or len(y) < 30:
            return []
        cols = list(X.columns)[:max_cols]
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(X), min(len(X), budget_rows), replace=False)
        S = X.iloc[idx][cols]
        M = np.zeros((len(S), len(cols)))
        disc = np.zeros(len(cols), bool)
        for j, c in enumerate(cols):
            if pd.api.types.is_numeric_dtype(S[c]):
                v = S[c].to_numpy(float)
                M[:, j] = np.where(np.isnan(v), np.nanmedian(v) if np.isfinite(v).any() else 0, v)
            else:
                M[:, j] = pd.factorize(S[c].fillna(MISSING_TOKEN))[0]
                disc[j] = True
        yy = np.asarray(y)[idx]
        if task == "classification":
            yy = pd.factorize(yy)[0]
            f = lambda t: mutual_info_classif(M, t, discrete_features=disc, random_state=SEED, n_neighbors=3)
        else:
            yy = yy.astype(float)
            f = lambda t: mutual_info_regression(M, t, discrete_features=disc, random_state=SEED, n_neighbors=3)
        mi = f(yy)
        null = f(rng.permutation(yy))
        thr = np.percentile(null, 90) * 1.5 + 0.005
        return [c for c, v in zip(cols, mi) if v <= thr]
    except Exception:
        return []


# =============================================================================== I/O
def read_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in (".json",):
        return pd.read_json(path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    # Never let automatic inference choose letters from a one-column header
    # as separators. Only common table delimiters are eligible.
    delimiter = "\t" if ext == ".tsv" else ","
    if ext != ".tsv":
        with open(path, "r", encoding="utf-8-sig", newline="") as stream:
            sample = stream.read(65536)
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            pass  # A genuine single-column CSV has no delimiter to detect.
    return pd.read_csv(path, sep=delimiter)


def write_predictions(out_dir, ids, labels, proba, classes, task, id_name="row_id"):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame({id_name: ids, "cluster" if task == "clustering" else "prediction": labels})
    if proba is not None:
        for i, c in enumerate(classes):
            df[f"proba_{c}"] = np.round(proba[:, i], 6)
    df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)


# =============================================================================== orchestration
def infer_task(y):
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "classification"
    nun = y.nunique(dropna=True)
    integral = np.allclose(y.dropna() % 1, 0)
    if integral and nun <= max(15, 0.02 * len(y)):
        return "classification"
    return "regression"


def solve(train, test=None, target=None, task=None, n_clusters=None, id_col=None, time_budget=40.0, checkpoint=None):
    budget = Budget(time_budget)
    if len(train) == 0:
        raise ValueError("training data is empty")
    info = {}
    if task is None:
        task = "clustering" if (n_clusters and target is None) else None
    y, target_name = None, None
    if task != "clustering":
        target_name = target or train.columns[-1]
        y = train[target_name]
        keep = y.notna().to_numpy()
        train = train.loc[keep].reset_index(drop=True)
        y = y.loc[keep].reset_index(drop=True)
        if len(y) == 0:
            raise ValueError("training target contains no observed labels")
        task = task or infer_task(y)
        if task == "classification" and pd.api.types.is_float_dtype(y) and np.allclose(y % 1, 0):
            y = y.astype(int)
        X = train.drop(columns=[target_name])
    else:
        X = train.copy()
    out_ids, out_id_name = None, "row_id"
    if id_col and id_col in X.columns:
        if test is None:
            out_ids, out_id_name = X[id_col].to_numpy(), id_col
        X = X.drop(columns=[id_col])
    if test is not None:
        if id_col and id_col in test.columns:
            out_ids, out_id_name = test[id_col].to_numpy(), id_col
            test = test.drop(columns=[id_col])
        if target_name and target_name in test.columns:
            test = test.drop(columns=[target_name])

    cleaner = FeatureCleaner().fit(X)
    Xc = cleaner.transform(X)

    if checkpoint is not None:
        Xt0 = Xc if test is None else cleaner.transform(test)
        fallback_info = {"chosen_models": {"prior": 1.0}, "checkpoint": True}
        probs0 = classes0 = None
        if task == "clustering":
            k0 = int(n_clusters or 0)
            total_rows = len(Xc) + (len(Xt0) if test is not None else 0)
            if not 1 <= k0 <= total_rows:
                raise ValueError("invalid n_clusters")
            labels0 = np.arange(total_rows) % k0
            if test is not None:
                labels0 = labels0[len(Xc):]
            fallback_info["note"] = "Emergency partition; no learned cluster structure"
        elif task == "classification":
            le0 = LabelEncoder().fit(y)
            labels0, probs0, classes0 = _prior_prediction(task, le0.transform(y), le0, len(le0.classes_), len(Xt0), fallback_info)
        else:
            labels0 = np.full(len(Xt0), np.mean(np.asarray(y, float)))
        ids0 = out_ids if out_ids is not None else np.arange(len(labels0))
        p0 = build_profile(X, cleaner, y, task, n_clusters, target_name, fallback_info, Xc, relevance=False)
        def emit0():
            fallback_info["runtime_seconds"] = round(budget.elapsed(), 2)
            checkpoint(dict(task=task, ids=ids0, id_name=out_id_name, labels=labels0,
                            proba=probs0, classes=classes0, profile=p0))
        emit0()
        if task != "clustering" and Xc.shape[1] and budget.select_open():
            try:
                from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
                model0 = DecisionTreeClassifier(max_depth=8, random_state=SEED) if task == "classification" else DecisionTreeRegressor(max_depth=8, random_state=SEED)
                fast0 = Pipeline([("prep", make_prep(Xc, "tree")), ("model", model0)])
                fast0.fit(Xc, y)
                labels0 = fast0.predict(Xt0)
                if task == "classification":
                    probs0, classes0 = fast0.predict_proba(Xt0), fast0.classes_
                fallback_info.update(chosen_models={"emergency_tree": 1.0}, note="Quick learned checkpoint before model search")
                emit0()
            except Exception:
                pass

    if task == "clustering":
        if not n_clusters:
            raise SystemExit("clustering needs --n-clusters")
        Xl = Xc if test is None else pd.concat([Xc, cleaner.transform(test)], ignore_index=True)
        labels = run_clustering(Xl, int(n_clusters), budget, info)
        if test is not None:
            labels = labels[len(Xc):]
        ids = out_ids if out_ids is not None else np.arange(len(labels))
        proba = classes = None
    else:
        Xt = Xc if test is None else cleaner.transform(test)
        labels, proba, classes = run_supervised(Xc, y, Xt, task, budget, info)
        ids = out_ids if out_ids is not None else np.arange(len(Xt))

    info["runtime_seconds"] = round(budget.elapsed(), 2)
    profile = build_profile(X, cleaner, y, task, n_clusters, target_name, info, Xc)
    info["runtime_seconds"] = round(budget.elapsed(), 2)
    cleaner.basic_ = None
    return dict(task=task, ids=ids, id_name=out_id_name, labels=labels, proba=proba, classes=classes,
                profile=profile)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="training table (or the table to cluster)")
    ap.add_argument("--test", default=None, help="table to predict (default: the training table)")
    ap.add_argument("--target", default=None, help="target column (default: last column)")
    ap.add_argument("--task", choices=["classification", "regression", "clustering"], default=None)
    ap.add_argument("--n-clusters", type=int, default=None)
    ap.add_argument("--id-col", default=None)
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--time-budget", type=float, default=40.0, help="soft budget in seconds (hard limit 60)")
    a = ap.parse_args(argv)

    train = read_table(a.train)
    test = read_table(a.test) if a.test else None
    checkpoint_count = 0
    def save(r):
        nonlocal checkpoint_count
        checkpoint_count += 1
        folder = os.path.join(a.out_dir, "checkpoint_" + str(checkpoint_count))
        write_predictions(folder, r["ids"], r["labels"], r["proba"], r["classes"], r["task"], r["id_name"])
        with open(os.path.join(folder, "profile.json"), "w") as f:
            json.dump(r["profile"], f, indent=2, default=str)
        marker = os.path.join(a.out_dir, "checkpoint.json")
        with open(marker + ".tmp", "w") as f:
            json.dump({"directory": os.path.basename(folder)}, f)
        os.replace(marker + ".tmp", marker)
    r = solve(train, test, a.target, a.task, a.n_clusters, a.id_col, a.time_budget, checkpoint=save)
    write_predictions(a.out_dir, r["ids"], r["labels"], r["proba"], r["classes"], r["task"], r["id_name"])
    with open(os.path.join(a.out_dir, "profile.json"), "w") as f:
        json.dump(r["profile"], f, indent=2, default=str)
    rep = r["profile"]["model_report"]
    print(json.dumps({"task": r["task"], "runtime_s": rep["runtime_seconds"], "models": rep.get("chosen_models"),
                      "cv": rep.get("blend_cv_score", rep.get("silhouette"))}))


if __name__ == "__main__":
    main()
