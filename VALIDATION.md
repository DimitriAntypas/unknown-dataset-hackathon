# Validation of the corrected submission

Validated on 24 October 2026. All four defects found in the submission review are corrected in main.py and in the self-contained Colab notebook.

## Corrections and regression checks

- Sorted integer measurements are preserved. The new end-to-end test requires at least 95% accuracy on unseen rows where a sorted integer measurement determines the target. Explicitly named ID columns are still removed.
- Boolean training columns accept native booleans, text forms of True and False, numeric 0 and 1, and missing markers in test data. An end-to-end test verifies successful output and the predictions for known boolean values.
- Delimiter detection only considers comma, semicolon, tab and pipe. Tests preserve single-column headers and values, quoted commas inside a single column, and all supported separators.
- Missing-value counts, fractions, cell totals and row totals are calculated from one normalized mask. Tests check datetime question marks and whitespace, numeric-text nulls and markers, nullable booleans, and overlapping missing cells in the same row.

## Test execution and environment

All 19 checks passed: seven original pipeline tests, four original robustness tests and eight new regression checks. They were run as the following scripts, without requiring pytest:

    python tests/test_pipeline.py
    python tests/test_fixes.py
    python tests/test_review_fixes.py

The deliberately stalled worker was terminated and its saved checkpoint recovered after approximately 8.02 seconds. Notebook code cells parse successfully and their embedded source files match the supplied project files.

Validation used Python 3.12.14, numpy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0, scipy 1.17.0.
This was the available runtime, not a fresh installation of the exact versions in requirements.txt. The dependency pins are preserved.

## Synthetic benchmark results

All eleven synthetic benchmarks completed successfully without supervisor timeout or fallback. Runtime measures the whole CLI process on this machine and is rounded to 0.1 seconds. These are local examples rather than official contest scores.

| Dataset | Task | Metric | Value | Wall seconds |
|---|---|---|---:|---:|
| bc | classification | accuracy | 0.952 | 4.5 |
| wine | classification | accuracy | 0.944 | 4.2 |
| imb | classification | accuracy | 0.940 | 7.0 |
| moons | classification | accuracy | 0.923 | 4.9 |
| diab | regression | r2 | 0.492 | 5.8 |
| fried | regression | r2 | 0.898 | 15.9 |
| bike | regression | r2 | 0.958 | 12.1 |
| iris | clustering | ari | 0.886 | 1.7 |
| blobs | clustering | ari | 0.997 | 2.2 |
| moons_c | clustering | ari | 0.496 | 1.5 |
| blobs5 | clustering | ari | 1.000 | 2.5 |

Machine-readable environment, test counts, benchmark metrics and model selections are included in validation_results.json. Reproduce the benchmarks with:

    python make_synthetic.py
    python evaluate.py

## Remaining submission requirements and limitations

- Replace the GitHub repository placeholder with a real public repository URL. A Colab link is optional unless separately requested.
- The supplied four team names are preserved. The brief specifies three students per team, so confirm an exception or update the team before submission.
- Run the official checker after adapting the entry command, metadata and output schema to the kickoff specification. Those materials and the nine official practice datasets were not supplied.
- Strongly non-convex clustering remains a performance limitation; the noisy two-moons result is reported above.
- Local timing does not certify timing on the assessment CPU or larger tables. The supervisor and checkpoint fallback remain in place.
- Earlier public-example measurements are retained in real_results.json as historical results from the uploaded package. Those CSVs were not included in this ZIP and were not rerun for this correction.
