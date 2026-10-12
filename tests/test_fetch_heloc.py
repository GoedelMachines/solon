"""fetch_heloc.py maps the Hugging Face mstz/heloc column names onto FICO's."""

import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("fetch_heloc", ROOT / "datasets/heloc/fetch_heloc.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mstz_columns_normalize_to_fico_layout(tmp_path):
    fetch = _load()
    columns = json.load(open(ROOT / "datasets/heloc/example_cases.json"))["columns"]
    assert sorted(fetch.MSTZ_COLUMNS.values()) == sorted(columns)
    row = list(range(1, 24)) + [1]
    csv = tmp_path / "mstz.csv"
    pd.DataFrame([row], columns=list(fetch.MSTZ_COLUMNS)).to_csv(csv, index=False)
    df = fetch.normalize(fetch.load_raw(str(csv)), columns)
    assert list(df.columns) == columns
    assert df.iloc[0].tolist() == row
