
# Reproducibility Checklist

This checklist provides instructions for reproducing the face verification pipeline on Windows, Linux, and macOS.

### Requirements

- Python 3.11 (recommended)
- pip
- Git
- Docker (optional)

The project uses TensorFlow 2.15.0 and related dependencies that require compatible Python and package versions. Python 3.11 is recommended for reproducing the original environment.

## 1. Clone the Repository

```bash
git clone https://github.com/gabebekele/face-verification-ml-system.git
cd face-verification-ml-system
```

## 2. Create a Virtual Environment

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

All subsequent Python commands can be run from either environment once the virtual environment is activated.

## 3. Install Dependencies

Install the full dependencies for data processing, evaluation, and inference:

```bash
python -m pip install -r requirements.txt
```

Alternatively, for inference-only environments that do not require dataset ingestion or evaluation, a lightweight dependency file is also available:

```bash
python -m pip install -r requirements-m3.txt
```

## 4. Run Dataset Ingestion

```bash
python scripts/ingest.py --config configs/m1.yaml
```

This generates the dataset manifest:

```text
outputs/manifest.json
```

### Windows: LFW Dataset Extraction

On Windows, TensorFlow Datasets may encounter a `[WinError 206]` error when extracting the LFW dataset because some generated file paths exceed Windows' traditional path-length limit.

If this happens, you can temporarily map your repository directory to a shorter drive path using PowerShell.

From the repository's original directory, run:

```powershell
$repo = (Get-Location).Path
subst X: "$repo"
Set-Location X:\
```

Then retry dataset ingestion:

```powershell
python scripts/ingest.py --config configs/m1.yaml
```

This allows TensorFlow Datasets to use shorter extraction paths without modifying the project's configuration or moving the repository.

The drive mapping is temporary and may need to be recreated after restarting Windows. Use an available drive letter if `X:` is already in use.

**Note:** This workaround is only needed on Windows when path-length errors occur. Linux and macOS users can follow the standard instructions.

**Docker note:** If Docker has trouble accessing the mapped X: drive, return to the repository's original directory before running Docker commands.

## 5. Generate Verification Pairs

```bash
python scripts/make_pairs.py --config configs/m1.yaml
```

This generates deterministic verification pairs for the train, validation, and test splits:

```text
outputs/pairs/train.csv
outputs/pairs/val.csv
outputs/pairs/test.csv
```

## 6. Run Performance Profiling

Run the profiling script using validation image pairs:

```bash
python scripts/profiling.py --config configs/m1.yaml --pairs outputs/pairs/val.csv --batch_sizes 1 5 10 25
```

The profiler measures preprocessing, embedding generation, similarity scoring, end-to-end latency, and throughput across different batch sizes.

The results can be compared with the [Profiling Report](Profiling%20report.pdf).

## 7. Run Face Verification Inference

Run inference on two images from the LFW dataset:

```bash
python scripts/run_inference.py --config configs/m1.yaml --left lfw/Zoran_Djindjic/Zoran_Djindjic_0001.jpg --right lfw/Zoran_Djindjic/Zoran_Djindjic_0002.jpg
```

The inference pipeline returns the cosine similarity score, verification prediction, confidence, and inference latency.

## 8. Run with Docker (Optional)

Docker provides an alternative way to run inference in a containerized environment.

**Build the Docker image:**

```bash
docker build -t face-verifier .
```

**Run inference:**

The following single-line command can be used in Linux/macOS terminals or Windows PowerShell:

```bash
docker run --rm -v "${PWD}:/app" face-verifier python scripts/run_inference.py --config configs/m1.yaml --left lfw/Zoran_Djindjic/Zoran_Djindjic_0001.jpg --right lfw/Zoran_Djindjic/Zoran_Djindjic_0002.jpg
```

Docker must be installed and running before executing these commands.

## 9. Run Automated Tests

Run the tests after completing dataset ingestion and verification pair generation:

```bash
python -m pytest
```
The project includes six automated tests covering evaluation metrics, threshold selection, validation logic, and pipeline integration.

All six tests passed during local reproducibility testing using Python 3.11 on Windows.

## 10. Final FaceNet Evaluation and Verification

The following scripts reproduce the final FaceNet evaluation results and verify threshold selection and dataset separation.

Run all commands from the project root after installing the dependencies from `requirements.txt` and preparing the LFW dataset and verification pairs.

### 10.1 Verify Threshold Selection

Run:

```bash
python scripts/verify_threshold.py
```

Expected results:

- Validation pairs: 400 (200 matching and 200 nonmatching)
- Selected threshold: 0.306122
- Validation balanced accuracy: 96.25%
- Threshold matches: True

The script evaluates 50 candidate thresholds between -1 and 1 using FaceNet similarity scores from the validation dataset. The selected threshold maximizes validation balanced accuracy.

### 10.2 Verify Dataset Separation

Run:

```bash
python scripts/check_data_leakage.py
```

Expected results:

- Train/validation shared identities: 0
- Train/test shared identities: 0
- Validation/test shared identities: 0
- Shared images across all splits: 0
- Incorrect pair labels: 0
- Incorrect split assignments: 0

These checks verify that the generated training, validation, and test pair datasets contain no overlapping identities or images.

### 10.3 Reproduce Final Model Evaluation

Run:

```bash
python scripts/evaluate_final.py
```

The script evaluates the final FaceNet model on all 4,046 held-out LFW test pairs using the validation-selected cosine similarity threshold.

Expected results:

| Metric | Result |
|---|---:|
| Total Test Pairs | 4,046 |
| Accuracy | 94.61% |
| Balanced Accuracy | 94.61% |
| Precision | 93.49% |
| Recall | 95.90% |
| F1 Score | 94.68% |
| ROC-AUC | 0.9886 |
| Decision Threshold | 0.306122 |

The confusion matrix contains:

- True Positives: 1,940
- True Negatives: 1,888
- False Positives: 135
- False Negatives: 83

Evaluation results are saved to `outputs/final_evaluation.json`.

These results demonstrate the final model's performance on the held-out LFW test pairs and provide reproducible checks for threshold selection and internal dataset separation.

## Final Release

The final system release is tagged:

```text
v2.2-final
```

To check out the tagged version:

```bash
git checkout v2.2-final
```

See the repository README for additional information about the system architecture, evaluation methodology, and limitations.

