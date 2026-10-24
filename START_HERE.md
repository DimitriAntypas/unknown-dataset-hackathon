# Start here

1. Extract the ZIP and open the folder containing `main.py` and `requirements.txt`.
2. Install Python 3.12. In a terminal inside that folder run:

   ```bash
   python -m venv .venv
   ```

   Activate it on Windows PowerShell with `.venv\Scripts\Activate.ps1`,
   or on macOS/Linux with `source .venv/bin/activate`.

3. Install and check:

   ```bash
   pip install -r requirements.txt
   python tests/test_pipeline.py
   python tests/test_fixes.py
   python tests/test_review_fixes.py
   python make_synthetic.py
   python evaluate.py
   ```

4. Upload or update the extracted project files in https://github.com/DimitriAntypas/unknown-dataset-hackathon.
   Keep `main.py`, `supervisor.py`, `requirements.txt`, `README.md`, all three test files,
   and the other included source files together. Include `.github/workflows/ci.yml`.
   Make the repository public. Do not upload virtual environments, generated outputs,
   credentials, or the downloaded Kaggle datasets.
5. All four supplied team names are retained. Confirm an exception to the brief's three-student team size
   before submitting, or update the names to match the approved team.
6. At kickoff, use the professor's exact command, metadata and output specification.
   Adapt this code if necessary and run the official checker on all nine official practice datasets.
   The package currently assumes the command-line and CSV/profile formats documented in README.md.
7. Submit https://github.com/DimitriAntypas/unknown-dataset-hackathon according to the professor's deadline instructions.

No website is required. The existing GitHub instructions PDF is a general guide;
this file and README.md describe the revised package, including its new supervisor module.

The supplied public datasets are optional additional tests. Extract their CSVs and run:

```bash
python validate_real.py --data-dir /path/to/csv-folder
```

The program does not download data or contact online services when it runs.
