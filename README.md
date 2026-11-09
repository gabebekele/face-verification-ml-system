# Face Verification ML System

An end-to-end face verification system built with Python and PyTorch that determines whether two facial images belong to the same person using pretrained FaceNet embeddings and cosine similarity.

The system uses the Labeled Faces in the Wild (LFW) dataset and InceptionResnetV1 pretrained on VGGFace2 to generate facial embeddings. A validation-selected similarity threshold is then used to classify image pairs as matching or non-matching identities.

Beyond model inference, the project emphasizes building a reproducible machine learning pipeline, including data processing, experimentation, model evaluation, automated testing, Docker-based inference, and performance profiling.

**Tech Stack:** Python, PyTorch, facenet-pytorch, NumPy, TensorFlow Datasets, Docker, pytest, YAML

## Requirements

- Python 3.11 (recommended)
- pip
- Git
- Docker (optional)

The project uses TensorFlow 2.15.0 and related dependencies that require compatible Python and package versions. Python 3.11 is recommended for reproducing the original environment.

## Key Features

- **Deterministic Data Processing:** Reproducible dataset preparation and identity-aware data splitting.

- **Face Embedding Generation:** Pretrained InceptionResnetV1 model for extracting facial features.

- **Similarity-Based Verification:** Cosine similarity scoring with validation-based threshold selection.

- **Reproducible Experimentation:** Configuration-driven experiments and automated evaluation.

- **Model Evaluation:** Performance metrics, ROC analysis, and error analysis.

- **Automated Testing:** Unit tests using pytest.

- **Docker-Based Inference:** Containerized environment for reproducible execution.

- **Performance Profiling:** CPU inference latency and throughput measurements.

## Key Results

The final face verification system uses pretrained InceptionResnetV1 embeddings and cosine similarity with a validation-selected operating threshold.

### Final System

| Metric | Result |
|---|---|
| Embedding Model | InceptionResnetV1 (VGGFace2) |
| Similarity Metric | Cosine Similarity |
| Operating Threshold | 0.3061 |
| Average CPU Inference Latency | 0.1312 seconds per pair |
| CPU Throughput | 7.62 pairs per second |

CPU performance was measured on an Intel Core i5-8350U using 25 validation image pairs. The reported latency and throughput represent a local CPU baseline.

### Earlier Baseline Experiments

Before implementing pretrained face embeddings, the project evaluated raw-pixel similarity using different normalization and threshold-selection strategies.

| Experiment | Balanced Accuracy |
|---|---|
| Raw-Pixel Baseline | 49.50% |
| Z-Score Normalization | 58.75% |

These experiments demonstrated the limitations of raw-pixel comparisons, particularly their sensitivity to differences in lighting, background, and facial appearance.

The final system replaces raw-pixel features with pretrained facial embeddings. The baseline accuracy values above do not represent the classification performance of the final FaceNet system.

## System Pipeline

```text
LFW Images
    |
    v
Image Loading & Preprocessing
    |
    v
InceptionResnetV1
Pretrained on VGGFace2
    |
    v
Face Embeddings
    |
    v
Cosine Similarity
    |
    v
Validation-Calibrated Threshold
    |
    v
Same Identity / Different Identity
```

## Methodology

### Data Processing

The project uses the **Labeled Faces in the Wild (LFW)** dataset.

The ingestion pipeline applies a fixed train, validation, and test split and generates a manifest describing the resulting dataset partitions.

Data processing is designed to be deterministic. Identities and filenames are sorted before splitting, and random operations use a fixed seed so that rerunning the pipeline with the same configuration produces consistent results.

Verification pairs are then generated for each split and stored as CSV files.

### Face Embeddings

The final verification system uses **InceptionResnetV1 pretrained on VGGFace2** through `facenet-pytorch`.

Each input face image is:

1. Loaded from the LFW dataset
2. Resized and normalized
3. Passed through InceptionResnetV1
4. Converted into a fixed-length embedding

The pretrained model is used as a feature extractor rather than training a face-recognition network from scratch.

### Similarity Scoring

The system compares two face embeddings using **cosine similarity**.

Higher similarity scores indicate that two face images are more likely to belong to the same person.

### Threshold Calibration

A verification system requires an operating threshold that converts the similarity score into a binary decision.

The final inference pipeline uses the validation-selected threshold:

```text
0.30612244897959173
```

The decision rule is:

```text
similarity >= threshold  -> Same Identity
similarity < threshold   -> Different Identity
```

Confidence is defined as the absolute distance between the similarity score and the operating threshold.

Predictions farther from the threshold are therefore treated as more confident than predictions close to the decision boundary.

## Evaluation and Error Analysis

The project includes a configuration-driven evaluation workflow for analyzing verification performance.

The evaluation system supports:

- Threshold sweeps
- Validation-based threshold selection
- Accuracy, precision, and recall reporting
- ROC curve generation
- False-positive analysis
- False-negative analysis
- Experiment-specific output directories
- Reproducible run configurations

Error cases are separated into false-positive and false-negative directories to support qualitative inspection of model failures.

This makes it possible to examine not only aggregate performance metrics, but also the kinds of examples that cause incorrect verification decisions.

## Performance Profiling

The final face verification system includes profiling tools to evaluate CPU inference performance across different batch sizes.

The profiler measures:

- Image preprocessing latency
- Face embedding generation latency
- Cosine similarity scoring latency
- Average end-to-end inference latency
- 95th-percentile (p95) latency
- Throughput in image pairs per second

### CPU Profiling Results

Performance was measured on an Intel Core i5-8350U CPU using validation image pairs from the LFW dataset.

| Batch Size | Avg. Latency (s/pair) | p95 Latency (s/pair) | Throughput (pairs/s) |
| ---------- | --------------------- | -------------------- | -------------------- |
| 1          | 1.0695                | 1.0695               | 0.93                 |
| 5          | 0.1398                | 0.1558               | 7.15                 |
| 10         | 0.1383                | 0.1559               | 7.23                 |
| 25         | 0.1312                | 0.1639               | 7.62                 |

### Key Findings

Embedding generation was the primary computational bottleneck, accounting for most of the inference time. Cosine similarity scoring contributed very little overhead.

Throughput improved as more image pairs were processed, reaching approximately **7.62 pairs per second**, with an average inference latency of **0.131 seconds per pair** in the 25-pair run.

The single-pair run was substantially slower, likely due to initialization or model warmup overhead. These results represent local CPU performance and may vary across hardware and execution environments.

The complete profiling methodology and latency breakdowns are available in the [Profiling Report](reports/Profiling%20report.pdf).

## Engineering and Reproducibility

The project is structured as a modular machine learning pipeline rather than a notebook-only experiment, with an emphasis on reproducibility, maintainability, and consistent evaluation.

Key engineering features include:

- **Modular Code:** Reusable Python modules and command-line scripts for data processing, evaluation, and inference.

- **Configuration Management:** YAML configuration files for managing experiment settings.

- **Deterministic Processing:** Fixed random seeds and identity-aware dataset splitting for reproducible results.

- **Experiment Tracking:** Organized evaluation outputs, including metrics, ROC curves, and error analysis artifacts.

- **Automated Testing:** Unit tests using pytest to validate evaluation functionality.

- **Containerization:** Docker support for reproducible inference environments.
  
A detailed reproducibility checklist is included [here](reports/reproducibility_checklist.md)

The final system version is tagged:

```text
v1.0-final
```

## Repository Structure

```text
face-verification-ml-system/
│
├── configs/                     # YAML experiment configurations
├── m1/                          # Supporting project files
├── reports/                     # System Card, profiling, and reports
├── scripts/                     # Data, inference, and profiling scripts
├── src/                         # Core Python package and evaluation logic
├── tests/                       # Automated tests
├── Dockerfile                   # Containerized inference environment
├── requirements.txt             # Full experimentation dependencies
├── requirements-m3.txt          # Lightweight inference dependencies
└── README.md
```

Generated datasets, outputs, and cached files are excluded from version control.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/gabebekele/face-verification-ml-system.git
cd face-verification-ml-system
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate the environment on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

For the full experimentation and evaluation environment:

```bash
pip install -r requirements.txt
```

A lightweight dependency environment is also provided for inference:

```bash
pip install -r requirements-m3.txt
```

## Reproduce the Data Pipeline

### Deterministic Dataset Ingestion

Run:

```bash
python scripts/ingest.py --config configs/m1.yaml
```

This generates:

```text
outputs/manifest.json
```

The manifest summarizes information including the dataset splits, counts, random seed, and split policy.

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

**Docker note:** If Docker has trouble accessing the mapped `X:` drive, return to the repository's original directory before running Docker commands.

### Generate Verification Pairs

Run:

```bash
python scripts/make_pairs.py --config configs/m1.yaml
```

This generates:

```text
outputs/pairs/train.csv
outputs/pairs/val.csv
outputs/pairs/test.csv
```

Each file contains positive and negative face-verification pairs for its respective dataset split.

### Run the Similarity Benchmark

Run:

```bash
python -m scripts.bench_similarity
```

The benchmark compares Python-loop and vectorized similarity implementations for both correctness and execution speed.

Results are saved to:

```text
outputs/bench_results.txt
```

## Reproduce Raw-Pixel Baseline Experiments

The evaluation framework supports configuration-driven experimental runs and command-line overrides.

To reproduce the selected Z-score experiment using balanced accuracy for threshold selection:

```bash
python -m src.evaluation --run_id run_3_zscore --norm z-score
```

This performs a threshold sweep, selects the operating decision boundary, and generates evaluation artifacts.

Additional experiments can be reproduced with the following commands.

### Baseline

```bash
python -m src.evaluation --run_id run_1_baseline
```

### F1 Threshold Selection

```bash
python -m src.evaluation --run_id run_2 --metric f1
```

### Min-Max Normalization

```bash
python -m src.evaluation --run_id run_4 --norm min_max
```

### Z-Score Normalization with F1 Selection

```bash
python -m src.evaluation --run_id run_5 --norm z-score --metric f1
```

Experiment outputs are stored under:

```text
outputs/runs/
```

Generated artifacts may include:

```text
run.json
roc_curve.png
errors/fp/
errors/fn/
```

## Run Face Verification Inference

To compare two face images locally:

```bash
python scripts/run_inference.py \
  --config configs/m1.yaml \
  --left lfw/<person_name>/<image_1>.jpg \
  --right lfw/<person_name>/<image_2>.jpg
```

Example:
```bash
python scripts/run_inference.py --config configs/m1.yaml --left lfw/Zoran_Djindjic/Zoran_Djindjic_0001.jpg --right lfw/Zoran_Djindjic/Zoran_Djindjic_0002.jpg
```

Image paths should be provided relative to the LFW dataset root using the format:

```text
lfw/<person_name>/<image_file>.jpg
```

The inference pipeline returns information including:

- Cosine similarity score
- Verification prediction
- Confidence
- Inference latency

## Run with Docker

### Build the Docker Image

```bash
docker build -t face-verifier .
```

### Run Inference in Docker

```bash
docker run --rm -v "${PWD}:/app" face-verifier \
  python scripts/run_inference.py \
  --config configs/m1.yaml \
  --left lfw/<person_name>/<image_1>.jpg \
  --right lfw/<person_name>/<image_2>.jpg
```

Example image paths:

```text
lfw/Zoran_Djindjic/Zoran_Djindjic_0001.jpg
lfw/Zoran_Djindjic/Zoran_Djindjic_0002.jpg
```

The Docker environment uses the lightweight inference dependency configuration to reduce unnecessary build complexity.

## Run Tests

The project includes automated tests for evaluation metrics, threshold selection, validation logic, and pipeline integration. The test suite contains six tests, all of which passed during local reproducibility testing using Python 3.11 on Windows.

Run the complete test suite:

```bash
python -m pytest
```

The integration test uses the configured LFW dataset root (`data/lfw`) and requires the dataset and verification pairs to be prepared beforehand.

## Run Performance Profiling

To reproduce profiling across multiple batch sizes:

```bash
python scripts/profiling.py --config configs/m1.yaml --pairs outputs/pairs/val.csv --batch_sizes 1 5 10 25
```

The profiling workflow reports latency breakdowns across the major stages of the inference pipeline.

Additional profiling documentation is available at:

```text
reports/Profiling report.pdf
```

## Outputs and Documentation

Important project artifacts are organized under the `outputs/` and `reports/` directories.

### Experiment Outputs

```text
outputs/
├── manifest.json
├── pairs/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── bench_results.txt
└── runs/
```

### Documentation

```text
reports/
├── M2_report.pdf
├── System_Card.pdf
├── Profiling report.pdf
└── reproducibility_checklist.md
```

The **System Card** describes the intended use of the system, assumptions, limitations, common failure cases, and potential fairness risks.

The **profiling report** documents preprocessing, embedding, scoring, and total latency behavior.

The **reproducibility checklist** provides the steps required to reproduce the system from a clean clone.

## Limitations

This system is an experimental machine learning project and is **not intended for production identity verification, authentication, surveillance, law enforcement, or other security-critical applications**.

Important limitations include:

- Performance depends on the characteristics and representation of the LFW dataset.
- The embedding model is pretrained on external facial datasets and may inherit limitations or biases from those datasets.
- Verification performance depends on the selected operating threshold.
- Lighting, pose, occlusion, camera quality, and other image conditions can affect embedding quality and similarity scores.
- Performance measured on LFW should not be assumed to generalize to other populations or deployment environments.
- Confidence represents distance from the decision threshold and should not be interpreted as a calibrated probability.
- Additional fairness, robustness, and deployment validation would be required before considering real-world use.

## Acknowledgments

This project originated from graduate machine learning coursework at the **University of Maryland**.
