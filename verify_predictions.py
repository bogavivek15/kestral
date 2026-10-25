"""Verify predictions.csv against the original sample submission and test order sequence."""
import argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="private_data")
args = parser.parse_args()
root = Path(__file__).resolve().parent
data_dir = Path(args.data_dir)
if not data_dir.is_absolute():
    data_dir = root / data_dir
needed = [data_dir / "test_unlabelled.csv", data_dir / "sample_submission.csv"]
missing = [str(path) for path in needed if not path.exists()]
if missing:
    raise SystemExit(
        "Cannot verify against the original assignment order list because these files are missing: "
        + ", ".join(missing)
        + ". Copy the private assignment CSVs into private_data/ first."
    )
predictions = pd.read_csv(root / "predictions.csv")
test = pd.read_csv(data_dir / "test_unlabelled.csv")
sample = pd.read_csv(data_dir / "sample_submission.csv")
assert list(predictions.columns) == ["order_id", "score"]
assert len(predictions) == len(test) == len(sample)
assert predictions["order_id"].astype(str).tolist() == test["order_id"].astype(str).tolist()
assert predictions["order_id"].astype(str).tolist() == sample["order_id"].astype(str).tolist()
assert predictions["order_id"].is_unique
assert predictions["score"].notna().all()
assert predictions["score"].between(0, 1).all()
print(f"PASS: {len(predictions):,} rows, exact schema/order, unique IDs, no missing scores, scores in [0, 1].")
