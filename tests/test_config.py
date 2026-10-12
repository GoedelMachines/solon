"""Tests for the config system: YAML loading, dataclass creation, CLI merging."""

import pytest
from solon.config import load_dataset_config, build_run_config, DatasetConfig, RunConfig


class TestLoadDatasetConfig:
    def test_loads_mini_config(self, heloc_yaml_path):
        cfg = load_dataset_config(heloc_yaml_path)
        assert isinstance(cfg, DatasetConfig)
        assert cfg.name == "heloc_mini"

    def test_target_fields(self, heloc_yaml_path):
        cfg = load_dataset_config(heloc_yaml_path)
        assert cfg.target.column == "RiskPerformance"
        assert cfg.target.positive_label == 0
        assert cfg.target.negative_label == 1

    def test_decisions(self, heloc_yaml_path):
        cfg = load_dataset_config(heloc_yaml_path)
        assert cfg.decisions.approve == "APPROVE"
        assert cfg.decisions.deny == "DENY"

    def test_special_values(self, heloc_yaml_path):
        cfg = load_dataset_config(heloc_yaml_path)
        assert cfg.special_values.values == [-7, -8, -9]
        assert cfg.special_values.meanings[-7] == "condition not met"

    def test_features(self, heloc_yaml_path):
        cfg = load_dataset_config(heloc_yaml_path)
        assert len(cfg.features) == 3
        assert cfg.features[0].name == "ExternalRiskEstimate"

    def test_feature_names_property(self, heloc_yaml_path):
        cfg = load_dataset_config(heloc_yaml_path)
        assert cfg.feature_names == [
            "ExternalRiskEstimate",
            "NetFractionRevolvingBurden",
            "NumTrades60Ever2DerogPubRec",
        ]

    def test_summary_features(self, heloc_yaml_path):
        cfg = load_dataset_config(heloc_yaml_path)
        assert "ExternalRiskEstimate" in cfg.summary_features

    def test_evaluation_weights(self, heloc_yaml_path):
        cfg = load_dataset_config(heloc_yaml_path)
        assert abs(sum(cfg.evaluation.weights.values()) - 1.0) < 0.01

    def test_defaults(self, heloc_yaml_path):
        cfg = load_dataset_config(heloc_yaml_path)
        assert cfg.defaults.sample_size == 10
        assert cfg.defaults.n_candidates == 2


class TestLoadFullHelocConfig:
    def test_loads_full_config(self, full_heloc_yaml_path):
        cfg = load_dataset_config(full_heloc_yaml_path)
        assert cfg.name == "heloc"
        assert len(cfg.features) == 23

    def test_all_features_populated(self, full_heloc_yaml_path):
        cfg = load_dataset_config(full_heloc_yaml_path)
        for f in cfg.features:
            assert f.name, f"Feature missing name: {f}"
            assert f.description, f"Feature {f.name} missing description"

    def test_regulatory_context_present(self, full_heloc_yaml_path):
        cfg = load_dataset_config(full_heloc_yaml_path)
        assert "FCRA" in cfg.regulatory_context
        assert "ECOA" in cfg.regulatory_context

    def test_summary_features_are_valid(self, full_heloc_yaml_path):
        cfg = load_dataset_config(full_heloc_yaml_path)
        names = cfg.feature_names
        for sf in cfg.summary_features:
            assert sf in names, f"Summary feature {sf} not in feature list"


class TestBuildRunConfig:
    def test_creates_run_config(self, heloc_yaml_path):
        rc = build_run_config(heloc_yaml_path, cases_csv="test.csv")
        assert isinstance(rc, RunConfig)
        assert rc.cases_csv == "test.csv"

    def test_defaults_from_dataset(self, heloc_yaml_path):
        rc = build_run_config(heloc_yaml_path, cases_csv="test.csv")
        assert rc.n_candidates == 2  # from mini config defaults
        assert rc.n_generations == 2
        assert rc.top_k == 1

    def test_cli_overrides(self, heloc_yaml_path):
        rc = build_run_config(
            heloc_yaml_path,
            cases_csv="test.csv",
            n_candidates=10,
            provider="openai",
        )
        assert rc.n_candidates == 10
        assert rc.provider == "openai"

    def test_none_overrides_ignored(self, heloc_yaml_path):
        rc = build_run_config(
            heloc_yaml_path,
            cases_csv="test.csv",
            n_candidates=None,
        )
        assert rc.n_candidates == 2  # still from defaults


class TestLoadLendingClubConfig:
    def test_loads_mini_config(self, lendingclub_yaml_path):
        cfg = load_dataset_config(lendingclub_yaml_path)
        assert isinstance(cfg, DatasetConfig)
        assert cfg.name == "lendingclub_mini"

    def test_target_fields(self, lendingclub_yaml_path):
        cfg = load_dataset_config(lendingclub_yaml_path)
        assert cfg.target.column == "loan_status"
        assert cfg.target.positive_label == "Fully Paid"
        assert cfg.target.negative_label == "Charged Off"

    def test_empty_special_values(self, lendingclub_yaml_path):
        cfg = load_dataset_config(lendingclub_yaml_path)
        assert cfg.special_values.values == []
        assert cfg.special_values.meanings == {}

    def test_mixed_feature_types(self, lendingclub_yaml_path):
        cfg = load_dataset_config(lendingclub_yaml_path)
        types = set(f.type for f in cfg.features)
        assert "numeric" in types
        assert "string" in types
        assert "categorical" in types

    def test_string_feature_present(self, lendingclub_yaml_path):
        cfg = load_dataset_config(lendingclub_yaml_path)
        emp = [f for f in cfg.features if f.name == "emp_title"][0]
        assert emp.type == "string"


class TestLoadFullLendingClubConfig:
    def test_loads_full_config(self, full_lendingclub_yaml_path):
        cfg = load_dataset_config(full_lendingclub_yaml_path)
        assert cfg.name == "lendingclub"
        assert len(cfg.features) == 61

    def test_all_features_populated(self, full_lendingclub_yaml_path):
        cfg = load_dataset_config(full_lendingclub_yaml_path)
        for f in cfg.features:
            assert f.name, f"Feature missing name: {f}"
            assert f.description, f"Feature {f.name} missing description"

    def test_summary_features_are_valid(self, full_lendingclub_yaml_path):
        cfg = load_dataset_config(full_lendingclub_yaml_path)
        names = cfg.feature_names
        for sf in cfg.summary_features:
            assert sf in names, f"Summary feature {sf} not in feature list"

    def test_string_labels(self, full_lendingclub_yaml_path):
        cfg = load_dataset_config(full_lendingclub_yaml_path)
        assert isinstance(cfg.target.positive_label, str)
        assert isinstance(cfg.target.negative_label, str)

    def test_bumped_simplicity_thresholds(self, full_lendingclub_yaml_path):
        cfg = load_dataset_config(full_lendingclub_yaml_path)
        assert cfg.evaluation.simplicity_thresholds["ideal_max"] == 15
        assert cfg.evaluation.simplicity_thresholds["good_max"] == 30
        assert cfg.evaluation.simplicity_thresholds["acceptable_max"] == 50
