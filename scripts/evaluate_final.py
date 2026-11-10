
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.embedding import embed_pair
from m1.similarity import cosine_similarity
from src.thresholding import apply_threshold


THRESHOLD = 0.30612244897959173


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/m1.yaml")
    parser.add_argument("--pairs", default="outputs/pairs/test.csv")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    with open(args.pairs, "r", newline="") as f:
        pairs = list(csv.DictReader(f))

    if args.limit is not None:
        pairs = pairs[:args.limit]

    if not pairs:
        raise ValueError("No test pairs found.")

    y_true = []
    y_pred = []
    scores = []

    for i, pair in enumerate(pairs, start=1):
        # Normalize Windows paths from the CSV
        left = pair["left_path"].replace("\\", "/")
        right = pair["right_path"].replace("\\", "/")
        label = int(pair["label"])

        # Reuse the existing FaceNet embedding pipeline
        left_vector, right_vector = embed_pair(
            left,
            right,
            config,
            normalization=None
        )

        score = float(cosine_similarity(
            np.asarray(left_vector).reshape(1, -1),
            np.asarray(right_vector).reshape(1, -1)
        )[0])

        prediction = int(apply_threshold(
            scores=[score],
            threshold=THRESHOLD,
            higher_is_same=True
        )[0])

        y_true.append(label)
        y_pred.append(prediction)
        scores.append(score)

        if i % 25 == 0 or i == len(pairs):
            print(f"Processed {i}/{len(pairs)} pairs")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    scores = np.array(scores)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    metrics = {
        "total_pairs": len(pairs),
        "positive_pairs": int(np.sum(y_true == 1)),
        "negative_pairs": int(np.sum(y_true == 0)),
        "threshold": THRESHOLD,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "roc_auc": (
            float(roc_auc_score(y_true, scores))
            if len(np.unique(y_true)) == 2
            else None
        ),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    filename = (
        "final_evaluation.json"
        if args.limit is None
        else "final_evaluation_preview.json"
    )

    output_path = output_dir / filename

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nFinal FaceNet Evaluation:")
    print(json.dumps(metrics, indent=4))
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
