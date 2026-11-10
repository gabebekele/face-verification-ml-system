
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.evaluation import (
    load_balanced_pairs,
    compute_similarity_scores,
    compute_metrics,
)

EXPECTED_THRESHOLD = 0.30612244897959173

with open(ROOT_DIR / "configs/m1.yaml", "r") as f:
    config = yaml.safe_load(f)

# Same validation sample used by the original evaluation
val_rows = load_balanced_pairs(
    config, "val", sample_per_class=200
)

val_scores, val_labels = compute_similarity_scores(
    val_rows, config, norm_type=None
)

thresholds = np.linspace(-1.0, 1.0, 50)

results = [
    compute_metrics(val_scores, val_labels, float(t))
    for t in thresholds
]

best = max(
    results,
    key=lambda x: x["balanced_accuracy"]
)

print("\nThreshold verification")
print("Validation pairs:", len(val_rows))
print("Selected threshold:", best["threshold"])
print("Validation balanced accuracy:", best["balanced_accuracy"])
print("Previously saved threshold:", EXPECTED_THRESHOLD)

matches = np.isclose(
    best["threshold"],
    EXPECTED_THRESHOLD,
    rtol=0,
    atol=1e-12
)

print("Threshold matches:", bool(matches))
