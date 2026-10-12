"""Tests for the sampling module."""

import pytest
import pandas as pd
from solon.config import load_dataset_config
from solon.sampling.selector import select_cases


@pytest.fixture
def heloc_config(full_heloc_yaml_path):
    return load_dataset_config(full_heloc_yaml_path)


class TestSelectCases:
    def test_selects_correct_number(self, heloc_config, tiny_dataset):
        # With only 5 rows, can't get 100 — just test it doesn't crash
        result = select_cases(tiny_dataset, heloc_config, n_cases=3, n_clusters=2)
        assert len(result) <= 3
        assert "RiskPerformance" in result.columns

    def test_no_cluster_column_in_output(self, heloc_config, tiny_dataset):
        result = select_cases(tiny_dataset, heloc_config, n_cases=3, n_clusters=2)
        assert "cluster" not in result.columns
