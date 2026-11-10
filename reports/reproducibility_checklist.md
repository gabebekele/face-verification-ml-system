
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


## Final Release

The final system release is tagged:

```text
v2.1-final
```

To check out the tagged version:

```bash
git checkout v2.1-final
```

See the repository README for additional information about the system architecture, evaluation methodology, and limitations.

