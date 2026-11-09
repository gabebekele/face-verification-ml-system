# Reproducibility Checklist

1. Clone repository
```text
git clone https://github.com/gabebekele/face-verification-ml-system.git
cd face-verification-ml-system
```

2. Create virtual environment
```text
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies
```text
pip install -r requirements-m3.txt
```

4. Run ingestion
```text
python scripts/ingest.py --config configs/m1.yaml
```

5. Generate pairs
```text
python scripts/make_pairs.py --config configs/m1.yaml
```

6. Run profiling
```text
python scripts/profiling.py --config configs/m1.yaml --pairs outputs/pairs/val.csv --batch_sizes 1 5 10 25
```

7. Run inference
```text
python scripts/run_inference.py --config configs/m1.yaml \
--left lfw/Zoran_Djindjic/Zoran_Djindjic_0001.jpg \
--right lfw/Zoran_Djindjic/Zoran_Djindjic_0002.jpg
```

8. Docker (optional)
```text
docker build -t face-verifier .

docker run --rm -v "${PWD}:/app" face-verifier \
python scripts/run_inference.py --config configs/m1.yaml \
--left lfw/Zoran_Djindjic/Zoran_Djindjic_0001.jpg \
--right lfw/Zoran_Djindjic/Zoran_Djindjic_0002.jpg
```

Final Tag
```text
v1.0-final
```
