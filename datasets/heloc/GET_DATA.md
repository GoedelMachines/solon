# Getting the HELOC data

The FICO HELOC dataset (10,459 applications, 23 features, `RiskPerformance` outcome; from the
FICO Explainable Machine Learning Challenge) is **not redistributed** here. Obtain it under its
own terms, then normalize it and rebuild the 100 cases the shipped examples refer to:

```bash
# from a CSV you downloaded (FICO heloc_dataset_v1.csv or the Kaggle copy):
python datasets/heloc/fetch_heloc.py --input path/to/heloc_dataset_v1.csv
# or from Hugging Face (needs `pip install datasets`):
python datasets/heloc/fetch_heloc.py
```

This writes `data/heloc_real.csv` and `data/heloc_100_cases.csv`. The cases are matched by
fingerprint (`example_cases.json`), so the targets in `examples/heloc/targets_claude-opus-4-6.json` line up
with them by `case_index`; pass `--cases data/heloc_100_cases.csv` wherever you reuse those
targets. To regenerate `examples/heloc/clustering/` (the full clustered CSV and few-shot cases
are not shipped):

```bash
solon cluster --dataset datasets/heloc/dataset.yaml --input data/heloc_real.csv --output data/clustering
```

No real data? `python datasets/heloc/generate_synthetic.py` writes `data/heloc_synthetic.csv`
with the same columns.
