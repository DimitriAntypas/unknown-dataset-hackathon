"""Optional validation on the three public example CSVs; no network downloads.
Usage: python validate_real.py --data-dir /path/to/extracted/csvs
These are examples, not the official nine practice datasets or official scores.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--report', default='real_results.json')
    a = ap.parse_args()
    root = Path(__file__).resolve().parent
    cases = [('Telco', 'WA_Fn-UseC_-Telco-Customer-Churn.csv', 'Churn', 'customerID'),
             ('Maintenance','ai4i2020.csv','Machine failure','UDI'),
             ('Mall','Mall_Customers.csv',None,'CustomerID')]
    results = []
    for name, filename, target, idcol in cases:
        matches = list(Path(a.data_dir).rglob(filename))
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one {filename}, found {len(matches)}')
        df = pd.read_csv(matches[0])
        if name == 'Maintenance':
            # Failure indicators reveal the target; do not use them as predictors.
            df = df.drop(columns=['TWF','HDF','PWF','OSF','RNF','Product ID'], errors='ignore')
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            command = [sys.executable, str(root/'main.py'), '--out-dir',str(t/'out'),'--id-col',idcol]
            truth = None
            if target:
                tr, te = train_test_split(df, test_size=.25, random_state=42, stratify=df[target])
                truth = te[target].reset_index(drop=True)
                tr.to_csv(t/'train.csv',index=False)
                te.drop(columns=target).to_csv(t/'test.csv',index=False)
                command += ['--train',str(t/'train.csv'),'--test',str(t/'test.csv'),'--target',target,'--task','classification']
            else:
                df.to_csv(t/'train.csv',index=False)
                command += ['--train',str(t/'train.csv'),'--task','clustering','--n-clusters','5']
            start = time.monotonic()
            proc = subprocess.run(command, capture_output=True, text=True, timeout=60)
            if proc.returncode:
                raise RuntimeError(proc.stderr)
            wall = time.monotonic()-start
            predictions = pd.read_csv(t/'out'/'predictions.csv')
            profile = json.loads((t/'out'/'profile.json').read_text())
            expected_ids = te[idcol].to_numpy() if target else df[idcol].to_numpy()
            assert (predictions[idcol].to_numpy() == expected_ids).all()
            result = {'dataset':name,'wall_seconds':round(wall,3),
                      'models':profile['model_report'].get('chosen_models'),
                      'timeout':profile['model_report']['supervisor_timeout']}
            if target:
                result.update(accuracy=accuracy_score(truth,predictions.prediction),
                    balanced_accuracy=balanced_accuracy_score(truth,predictions.prediction),
                    macro_f1=f1_score(truth,predictions.prediction,average='macro'))
                positive = 'Yes' if name == 'Telco' else 1
                result['roc_auc'] = roc_auc_score(truth == positive,predictions['proba_'+str(positive)])
            else:
                assert predictions.cluster.nunique() == 5
                result['clusters'] = 5
                result['note'] = 'Five clusters assumed for this example; no ground-truth labels supplied.'
            results.append(result)
            print(json.dumps(result), flush=True)
    Path(a.report).write_text(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
