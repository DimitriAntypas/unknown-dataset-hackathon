# Unknown Dataset Hackathon — submission

**Team:** Georgio Sleiman, Andrew Sleiman, Carla Doumit, Dimitri Antypas

**GitHub repository:** REPLACE_WITH_YOUR_PUBLIC_GITHUB_REPOSITORY_URL

**Google Colab notebook:** REPLACE_WITH_YOUR_SHARED_COLAB_NOTEBOOK_URL

Replace the GitHub placeholder before submission. A shared Colab link is optional unless separately requested.
No live repository or Colab notebook has been created by this package. The brief specifies three students per team;
confirm an exception before submitting the four names above.

- `Unknown_Dataset_Hackathon.ipynb`: self-contained Google Colab notebook. Upload it to Colab and run the numbered cells.
- `main.py` and `supervisor.py`: command-line entry point and timeout supervisor; keep both at the repository root.
- `requirements.txt`: pinned dependencies.
- `UPLOAD_INSTRUCTIONS.md`: exact upload and submission steps.
- `VALIDATION.md`: test results and known limitations.


# Unknown Dataset Hackathon – automatic ML for unfamiliar tables

**Team:** Georgio Sleiman, Andrew Sleiman, Carla Doumit, Dimitri Antypas  |  **Entry point:** `python main.py`  |  **Python:** 3.12 (3.11+ required by the pinned versions)

One program that inspects an unfamiliar table, chooses its own preparation and model, and writes predictions plus a
structured data profile. It covers **classification, regression and clustering** with no per-dataset configuration.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py --train train.csv --test test.csv --target y --out-dir out                 # supervised
python main.py --train data.csv --task clustering --n-clusters 4 --out-dir out            # clustering
```
| flag | meaning |
|---|---|
| `--train` | training table (or the table to cluster) |
| `--test` | table to predict (defaults to the training table) |
| `--target` | target column (default: last column) |
| `--task` | `classification` / `regression` / `clustering` (inferred from the target if omitted) |
| `--n-clusters` | required for clustering |
| `--id-col` | column copied into the output under its original name |
| `--time-budget` | search budget in seconds (default 40); CLI worker cutoff is min(budget + 8, 52) |

Outputs: `out/predictions.csv` (`row_id`, or the supplied ID column name, `prediction` or `cluster`, plus `proba_<class>` columns) and
`out/profile.json`.

## How it works
1. **Cleaning (`FeatureCleaner`)** – learns types on the training table and applies the same plan to any test table:
   numbers stored as text (e.g. `" "` blanks), datetime strings (expanded to year/month/day/weekday/hour/epoch),
   booleans, `?`/blank missing markers, infinities; drops constant, empty, exact-duplicate, perfectly-correlated
   (scaled copies) and ID-like columns; frequency-encodes high-cardinality categoricals; tolerates missing columns at
   test time.
2. **Model-specific preprocessing** – native-NaN passthrough for gradient boosting, median imputation (+ missing
   indicators) for forests, imputation + signed-log skew fix + scaling for linear / neural / kNN models; one-hot with
   rare-level grouping and unseen-level handling.
3. **Supervised search** – cross-validated out-of-fold predictions for up to 9 candidates (two gradient-boosting
   variants, log-target boosting for skewed targets, random forest, extra trees, linear/logistic, polynomial ridge /
   logistic, MLP, kNN). A **greedy ensemble selection** on the out-of-fold predictions (log-loss / RMSE) picks the blend
   weights; chosen models are refit on all rows. Imbalanced classification then tunes class prior weights on the
   out-of-fold probabilities against a hedge of accuracy, balanced accuracy and macro-F1.
4. **Clustering** – several feature spaces (all numeric, a relevance-filtered subset, one-hot mix, PCA) with model-based
   imputation × k-means, Gaussian mixtures (spherical/diag/full), Ward, spectral. The winner is chosen by silhouette,
   consensus with the other labelings and nearest-neighbour purity, with penalties for empty/tiny clusters.
5. **Runtime supervision** – the CLI starts a lightweight parent process before importing scientific libraries.
   The worker saves a complete prior checkpoint and, for supervised data, a quick learned tree checkpoint before
   the longer search. At `min(time_budget + 8, 52)` seconds the parent kills a stalled worker and publishes its latest
   complete checkpoint. The 52-second cap leaves headroom below the contest's 60-second limit. This is a tested
   safeguard, not a guarantee on arbitrary hardware or unbounded tables; no checkpoint may be available if initial
   reading/cleaning itself exhausts the allowance. Direct Python `solve()` calls have a fresh soft budget each time;
   use the CLI for process-level timeout protection. The CLI fixes numerical-library and forest concurrency to one thread.
6. **Profile (`profile.json`)** – task, shape, column types (incl. text-numbers, datetimes), missing counts/fractions,
   constant / duplicate / perfectly-correlated / ID-like columns, duplicate rows, cardinality, scale ratio and
   `differently_scaled_features`, skewed columns, outlier counts, class distribution and `imbalance_ratio`, target
   statistics, `likely_irrelevant_features` (mutual information vs a shuffled-target baseline), non-linear-signal flag,
   plus the model report (CV scores, blend weights, timings).

## Validation and reproducibility

Run the complete checks in a fresh environment:
```bash
pip install -r requirements.txt
pip install pytest
python -m pytest -q
python make_synthetic.py
python evaluate.py
```
Without pytest, run `python tests/test_pipeline.py`, `python tests/test_fixes.py`, and
`python tests/test_review_fixes.py`.
The benchmark runner now returns failure if any dataset crashes or exceeds its 60-second subprocess limit.
A successful benchmark run confirms execution; it does not enforce an official quality threshold.

For the three supplied public examples, extract their CSVs into one data directory and run:
```bash
python validate_real.py --data-dir /path/to/data --report real_results.json
```
This uses a fixed stratified 75/25 holdout for classification. For predictive maintenance, the five failure-type
indicators and Product ID are excluded to avoid an unrepresentative test using outcome-related fields.
Mall uses five clusters as a benchmark assumption; no ground-truth cluster labels are provided.
See `VALIDATION.md` and `real_results.json` for measured results and limitations.
These public examples and generated synthetic tables are not the nine official kickoff datasets.

## Assumptions (reviewed at kickoff)
- Input is CSV (comma, semicolon, tab and pipe delimiters are supported, as is JSON; Parquet/Excel need `pyarrow`/`openpyxl`, which are not pinned); the target is `--target` or the last column.
- Output is `row_id, prediction` (or `cluster`). If the official contract differs, adapt the CLI, metadata parsing, `read_table`, `write_predictions`, and profile schema as necessary. The official checker has not been provided.
- The dataset states its task; inferring it from the target is only a fallback.
- CLI uses one thread, no network at run time, and fixed random seeds. Time-dependent model selection can still vary with machine speed and load.
- The scoring metric for classification is not known in advance, so imbalanced problems are tuned for a blend of
  accuracy, balanced accuracy and macro-F1.

## Fixes in this revision
- Sorted, unique integer measurements are retained unless their name identifies them as an ID.
- Boolean features accept native values, boolean strings and missing markers consistently in test data.
- Delimiter detection is restricted to comma, semicolon, tab and pipe, preserving single-column CSVs.
- Missing counts, fractions, cell totals and row totals use one mask, including datetime missing markers.
- Each `solve()` invocation starts a new monotonic budget.
- CLI worker timeout recovery publishes a completed checkpoint, with `supervisor_timeout` recorded.
- Clustering validates k and handles k=1, k=n, constant data, and collapsed duplicate-point clusters.
  On data with no separating information, forced groups are deterministic partitions, not evidence of real structure.
- Missing-value row counts include both blank and `?` markers.
- ID values are retained even when no separate test table is supplied.
- Empty training data and entirely missing targets produce clear errors.
- Tests cover these cases and a deliberately stalled worker.

## Known limitations
- Clustering selection favours compact (convex) groups; strongly non-convex shapes (e.g. two moons) may be split
  incorrectly.
- Very large tables rely on subsampling for model selection (final fits use every row).

## Tests
`python -m pytest -q` (or the three test scripts above without pytest) covers string labels, missing test columns,
text-numbers, datetimes, constant/duplicate/ID columns, tiny and single-class data, high cardinality, clustering with
missing values, and the CLI end to end. The eight checks in `tests/test_review_fixes.py` specifically cover the
four corrected defects, including prediction quality on unseen sorted integer measurements, boolean test
columns with missing markers, quoted single-column CSVs, supported delimiters and datetime missing counts.
