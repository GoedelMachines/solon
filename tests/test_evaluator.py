"""Tests for the evaluator module."""

import pytest
import pandas as pd
from solon.config import load_dataset_config
from solon.pipeline.evaluator import evaluate_candidate, count_conditions, rank_candidates


@pytest.fixture
def heloc_config(full_heloc_yaml_path):
    return load_dataset_config(full_heloc_yaml_path)


class TestCountConditions:
    def test_simple_code(self):
        code = "if x > 5:\n  pass\nelif x > 3:\n  pass\nelse:\n  pass"
        assert count_conditions(code) == 2

    def test_no_conditions(self):
        assert count_conditions("x = 5\nreturn x") == 0


class TestEvaluateCandidate:
    def test_valid_program(self, heloc_config, sample_program_code, sample_targets, tiny_dataset):
        result = evaluate_candidate(
            code=sample_program_code,
            targets=sample_targets,
            cases_df=tiny_dataset,
            config=heloc_config,
        )
        assert result["valid"] is True
        assert 0 <= result["overall_score"] <= 1.0
        assert "accuracy" in result["metrics"]
        assert "simplicity" in result["metrics"]
        assert "robustness" in result["metrics"]
        assert "coverage" in result["metrics"]

    def test_invalid_program(self, heloc_config, sample_targets, tiny_dataset):
        result = evaluate_candidate(
            code="def underwrite(x): raise Exception('boom')",
            targets=sample_targets,
            cases_df=tiny_dataset,
            config=heloc_config,
        )
        # Still loads but may crash on execution
        assert result["valid"] is True

    def test_syntax_error_program(self, heloc_config, sample_targets, tiny_dataset):
        result = evaluate_candidate(
            code="def underwrite(x incomplete syntax",
            targets=sample_targets,
            cases_df=tiny_dataset,
            config=heloc_config,
        )
        assert result["valid"] is False
        assert result["hard_constraint_pass"] is False

    def test_misclassifications_use_summary_features(self, heloc_config, sample_program_code, sample_targets, tiny_dataset):
        result = evaluate_candidate(
            code=sample_program_code,
            targets=sample_targets,
            cases_df=tiny_dataset,
            config=heloc_config,
        )
        if result["misclassifications"]:
            m = result["misclassifications"][0]
            # Check that features come from config.summary_features
            for key in m["features"]:
                assert key in heloc_config.summary_features

    def test_with_holdout(self, heloc_config, sample_program_code, sample_targets, tiny_dataset):
        result = evaluate_candidate(
            code=sample_program_code,
            targets=sample_targets,
            cases_df=tiny_dataset,
            config=heloc_config,
            holdout_df=tiny_dataset,
        )
        assert "holdout_approval_precision" in result["metrics"]
        assert "holdout_denial_precision" in result["metrics"]


class TestRankCandidates:
    def test_ranking(self):
        evals = [
            {"overall_score": 0.5},
            {"overall_score": 0.9},
            {"overall_score": 0.3},
        ]
        ranking = rank_candidates(evals)
        assert ranking == [1, 0, 2]
