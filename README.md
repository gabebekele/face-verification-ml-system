# Face Verification ML System

A reproducible face verification pipeline built with **Python and PyTorch** using pretrained InceptionResnetV1 embeddings, cosine similarity, validation-based threshold calibration, automated evaluation, Docker-containerized inference, and latency profiling.

## Overview

This project implements an end-to-end face verification system using the **Labeled Faces in the Wild (LFW)** dataset.

The system generates facial embeddings using **InceptionResnetV1 pretrained on VGGFace2**, compares pairs of embeddings using cosine similarity, and applies a validation-selected threshold to determine whether two images represent the same identity.

The project focuses on the broader machine learning lifecycle rather than model inference alone, including:

- Deterministic data processing
- Reproducible experimentation
- Validation-based threshold selection
- Model evaluation and error analysis
- Automated testing
- Docker-based inference
- Latency and throughput profiling
- System and reproducibility documentation

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

## Key Features

- End-to-end face verification workflow
- Pretrained deep-learning face embeddings
- Cosine-similarity verification
- Validation-based operating-threshold selection
- Deterministic dataset processing
- Identity-aware train, validation, and test splits
- Configuration-driven experiments using YAML
- Vectorized similarity computation
- Automated false-positive and false-negative analysis
- Unit testing with `pytest`
- Docker-containerized inference
- Batch-size, latency, and throughput profiling
- System Card documenting intended use, limitations, and potential risks
- Reproducibility checklist for running the project from a clean environment

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

The final system includes profiling tools for measuring inference performance across different batch sizes.

The profiler measures:

- Image preprocessing time
- Embedding-generation time
- Similarity-scoring time
- Total inference latency
- Performance changes across batch sizes

This provides a baseline for understanding the computational behavior of the verification pipeline and identifying potential bottlenecks.

## Engineering and Reproducibility

The project is organized as a reusable ML pipeline rather than a notebook-only experiment.

Engineering and reproducibility features include:

- Modular Python source code
- YAML-based configuration
- Fixed random seeds
- Deterministic data handling
- Automated tests
- Reusable command-line scripts
- Experiment-specific artifacts
- Docker support
- Performance profiling
- System documentation
- Reproducibility documentation

A reproducibility checklist is included at:

```text
reports/reproducibility_checklist.md
```

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

## Technologies

### Machine Learning

- Python
- PyTorch
- `facenet-pytorch`
- InceptionResnetV1
- VGGFace2 pretrained weights
- Face embeddings
- Cosine similarity

### Data and Evaluation

- NumPy
- Labeled Faces in the Wild
- `tensorflow_datasets`
- Vectorized similarity computation
- Threshold calibration
- Error analysis
- Performance profiling

### Engineering

- Docker
- `pytest`
- YAML
- Git
- GitHub
- Command-line tooling

# How to Run

## 1. Clone the Repository

```bash
git clone https://github.com/gabebekele/face-verification-ml-system.git
cd face-verification-ml-system
```

## 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate the environment on macOS or Linux:

```bash
source .venv/bin/activate
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

For the full experimentation and evaluation environment:

```bash
pip install -r requirements.txt
```

A lightweight dependency environment is also provided for inference:

```bash
pip install -r requirements-m3.txt
```

# Reproduce the Data Pipeline

## Deterministic Dataset Ingestion

Run:

```bash
python scripts/ingest.py --config configs/m1.yaml
```

This generates:

```text
outputs/manifest.json
```

The manifest summarizes information including the dataset splits, counts, random seed, and split policy.

## Generate Verification Pairs

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

## Run the Similarity Benchmark

Run:

```bash
python -m scripts.bench_similarity
```

The benchmark compares Python-loop and vectorized similarity implementations for both correctness and execution speed.

Results are saved to:

```text
outputs/bench_results.txt
```

# Run Model Evaluation

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

# Run Face Verification Inference

To compare two face images locally:

```bash
python scripts/run_inference.py \
  --config configs/m1.yaml \
  --left lfw/<person_name>/<image_1>.jpg \
  --right lfw/<person_name>/<image_2>.jpg
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

# Run with Docker

## Build the Docker Image

```bash
docker build -t face-verifier .
```

## Run Inference in Docker

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

# Run Tests

Run the evaluation metric tests:

```bash
pytest tests/test_metrics.py
```

To run the complete available test suite:

```bash
pytest
```

# Run Performance Profiling

To reproduce profiling across multiple batch sizes:

```bash
python scripts/profiling.py \
  --config configs/m1.yaml \
  --pairs outputs/pairs/val.csv \
  --batch_sizes 1 5 10 25
```

The profiling workflow reports latency breakdowns across the major stages of the inference pipeline.

Additional profiling documentation is available at:

```text
reports/Profiling report.pdf
```

# Outputs and Documentation

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

# Limitations

This system is an experimental machine learning project and is **not intended for production identity verification, authentication, surveillance, law enforcement, or other security-critical applications**.

Important limitations include:

- Performance depends on the characteristics and representation of the LFW dataset.
- The embedding model is pretrained on external facial datasets and may inherit limitations or biases from those datasets.
- Verification performance depends on the selected operating threshold.
- Lighting, pose, occlusion, camera quality, and other image conditions can affect embedding quality and similarity scores.
- Performance measured on LFW should not be assumed to generalize to other populations or deployment environments.
- Confidence represents distance from the decision threshold and should not be interpreted as a calibrated probability.
- Additional fairness, robustness, and deployment validation would be required before considering real-world use.


# What This Project Demonstrates

This project demonstrates experience with:

- Building reproducible machine learning pipelines
- Working with pretrained deep-learning models
- Computer vision and face embeddings
- Embedding-based inference
- Validation-based threshold calibration
- Model evaluation and error analysis
- Deterministic data engineering
- Configuration-driven experimentation
- Automated testing
- Docker-based inference
- Latency and throughput profiling
- Technical ML documentation
- Production-oriented machine learning engineering practices

The project reinforced the importance of treating machine learning as a **complete system engineering problem**, rather than focusing only on model development or a single performance metric.

## Acknowledgments

This project originated from graduate machine learning coursework at the **University of Maryland**.