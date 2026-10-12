"""Shared test fixtures."""

import json
import os
import pytest
import pandas as pd

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def fixtures_dir():
    return FIXTURES_DIR


@pytest.fixture
def heloc_yaml_path():
    return os.path.join(FIXTURES_DIR, "heloc_dataset.yaml")


@pytest.fixture
def full_heloc_yaml_path():
    return os.path.join(
        os.path.dirname(__file__), "..", "datasets", "heloc", "dataset.yaml"
    )


@pytest.fixture
def tiny_dataset():
    return pd.read_csv(os.path.join(FIXTURES_DIR, "tiny_dataset.csv"))


@pytest.fixture
def sample_targets():
    with open(os.path.join(FIXTURES_DIR, "sample_targets.json")) as f:
        return json.load(f)


@pytest.fixture
def sample_program_code():
    with open(os.path.join(FIXTURES_DIR, "sample_program.py")) as f:
        return f.read()


@pytest.fixture
def lendingclub_yaml_path():
    return os.path.join(FIXTURES_DIR, "lendingclub_dataset.yaml")


@pytest.fixture
def full_lendingclub_yaml_path():
    return os.path.join(
        os.path.dirname(__file__), "..", "datasets", "lendingclub", "dataset.yaml"
    )


@pytest.fixture
def tiny_lendingclub_dataset():
    return pd.read_csv(os.path.join(FIXTURES_DIR, "tiny_lendingclub.csv"))
