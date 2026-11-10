
import csv
from pathlib import PureWindowsPath

splits = {}

for split in ["train", "val", "test"]:
    identities = set()
    images = set()
    incorrect_labels = 0
    incorrect_splits = 0
    pair_count = 0

    with open(f"outputs/pairs/{split}.csv", newline="") as f:
        for row in csv.DictReader(f):
            left = PureWindowsPath(row["left_path"])
            right = PureWindowsPath(row["right_path"])

            left_id = left.parent.name
            right_id = right.parent.name

            identities.update([left_id, right_id])
            images.update([str(left), str(right)])

            expected_label = int(left_id == right_id)

            if int(row["label"]) != expected_label:
                incorrect_labels += 1

            if row["split"] != split:
                incorrect_splits += 1

            pair_count += 1

    splits[split] = {
        "identities": identities,
        "images": images
    }

    print(f"\n{split.upper()}")
    print(f"Pairs: {pair_count}")
    print(f"Unique identities: {len(identities)}")
    print(f"Unique images: {len(images)}")
    print(f"Incorrect labels: {incorrect_labels}")
    print(f"Incorrect split values: {incorrect_splits}")


print("\nCROSS-SPLIT OVERLAP")

comparisons = [
    ("train", "val"),
    ("train", "test"),
    ("val", "test")
]

for a, b in comparisons:
    identity_overlap = (
        splits[a]["identities"] & splits[b]["identities"]
    )

    image_overlap = (
        splits[a]["images"] & splits[b]["images"]
    )

    print(f"\n{a} vs {b}")
    print(f"Shared identities: {len(identity_overlap)}")
    print(f"Shared images: {len(image_overlap)}")

    if identity_overlap:
        print("Examples:", sorted(identity_overlap)[:5])
