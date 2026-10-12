"""Tests for prompt templates and formatters."""

import pandas as pd
import pytest
from solon.config import load_dataset_config
from solon.prompts.templates import (
    render_target_system_prompt,
    render_synthesis_system_prompt,
    render_program_skeleton,
)
from solon.prompts.formatters import format_case, build_batch_prompt, build_synthesis_prompt


class TestTargetSystemPrompt:
    def test_renders_heloc(self, full_heloc_yaml_path):
        cfg = load_dataset_config(full_heloc_yaml_path)
        prompt = render_target_system_prompt(cfg)
        assert "APPROVE" in prompt
        assert "DENY" in prompt
        assert "FCRA" in prompt or "regulatory" in prompt.lower()
        assert "ExternalRiskEstimate" in prompt
        assert len(prompt) > 500

    def test_renders_mini(self, heloc_yaml_path):
        cfg = load_dataset_config(heloc_yaml_path)
        prompt = render_target_system_prompt(cfg)
        assert "APPROVE" in prompt
        assert "DENY" in prompt
        assert "ExternalRiskEstimate" in prompt

    def test_special_values_in_prompt(self, full_heloc_yaml_path):
        cfg = load_dataset_config(full_heloc_yaml_path)
        prompt = render_target_system_prompt(cfg)
        assert "-7" in prompt
        assert "-8" in prompt
        assert "-9" in prompt


class TestSynthesisSystemPrompt:
    def test_renders_heloc(self, full_heloc_yaml_path):
        cfg = load_dataset_config(full_heloc_yaml_path)
        prompt = render_synthesis_system_prompt(cfg)
        assert "def underwrite" in prompt
        assert "APPROVE" in prompt
        assert "DENY" in prompt
        assert "ExternalRiskEstimate" in prompt

    def test_renders_mini(self, heloc_yaml_path):
        cfg = load_dataset_config(heloc_yaml_path)
        prompt = render_synthesis_system_prompt(cfg)
        assert "def underwrite" in prompt


class TestProgramSkeleton:
    def test_renders(self, full_heloc_yaml_path):
        cfg = load_dataset_config(full_heloc_yaml_path)
        skeleton = render_program_skeleton(cfg)
        assert "def underwrite" in skeleton
        assert "def get(feature" in skeleton
        assert "-7, -8, -9" in skeleton


class TestFormatCase:
    def test_format_case(self, full_heloc_yaml_path, tiny_dataset):
        cfg = load_dataset_config(full_heloc_yaml_path)
        row = tiny_dataset.iloc[0]
        text = format_case(row, cfg)
        assert "APPLICATION DATA:" in text
        assert "ExternalRiskEstimate: 82" in text
        # RiskPerformance should NOT appear
        assert "RiskPerformance" not in text

    def test_special_values_labeled(self, full_heloc_yaml_path, tiny_dataset):
        cfg = load_dataset_config(full_heloc_yaml_path)
        # Row 1 has MSinceMostRecentDelq = -7
        row = tiny_dataset.iloc[1]
        text = format_case(row, cfg)
        assert "condition not met" in text


class TestBuildBatchPrompt:
    def test_batch_prompt(self, full_heloc_yaml_path, tiny_dataset):
        cfg = load_dataset_config(full_heloc_yaml_path)
        cases = [(i, tiny_dataset.iloc[i]) for i in range(3)]
        prompt = build_batch_prompt(cases, cfg)
        assert "APPLICATION 0" in prompt
        assert "APPLICATION 1" in prompt
        assert "APPLICATION 2" in prompt
        assert "JSON array" in prompt


class TestBuildSynthesisPrompt:
    def test_synthesis_prompt(self, full_heloc_yaml_path, sample_targets):
        cfg = load_dataset_config(full_heloc_yaml_path)
        prompt = build_synthesis_prompt(sample_targets, cfg)
        assert "APPROVE" in prompt or "DENY" in prompt
        assert "underwrite()" in prompt

    def test_with_previous_best(self, full_heloc_yaml_path, sample_targets):
        cfg = load_dataset_config(full_heloc_yaml_path)
        prev = [{
            "code": "def underwrite(app): pass",
            "score": 0.75,
            "metrics": {"accuracy": 0.8, "correct": 4, "total_compared": 5,
                        "simplicity": 0.9, "n_conditions": 5, "robustness": 0.7,
                        "coverage": 0.9},
            "misclassifications": [{
                "case_index": 1,
                "predicted": "APPROVE",
                "expected": "DENY",
                "features": {"ExternalRiskEstimate": 65, "NetFractionRevolvingBurden": 80,
                              "NumTrades60Ever2DerogPubRec": 3},
            }],
        }]
        prompt = build_synthesis_prompt(sample_targets, cfg, previous_best=prev)
        assert "PREVIOUS BEST" in prompt
        assert "MISCLASSIFIED" in prompt
        assert "ExternalRiskEstimate=65" in prompt


class TestEmptySpecialValues:
    """Templates must work when special_values.values is empty (e.g., Lending Club)."""

    def test_skeleton_no_special_values(self):
        """get() helper must be valid Python when no special values exist."""
        from solon.config.schema import DatasetConfig, SpecialValueSpec, TargetSpec
        cfg = DatasetConfig(
            name="test",
            target=TargetSpec(column="label"),
            special_values=SpecialValueSpec(values=[], meanings={}, treatment="missing"),
        )
        skeleton = render_program_skeleton(cfg)
        compile(skeleton, "<skeleton>", "exec")
        assert "(,)" not in skeleton

    def test_synthesis_prompt_no_special_values(self):
        """Synthesis prompt must not mention 'Special values []'."""
        from solon.config.schema import (
            DatasetConfig, SpecialValueSpec, FeatureSpec, SynthesizerConfig, TargetSpec
        )
        cfg = DatasetConfig(
            name="test",
            target=TargetSpec(column="label"),
            special_values=SpecialValueSpec(values=[], meanings={}, treatment="missing"),
            features=[FeatureSpec(name="loan_amnt", description="Loan amount", type="numeric")],
            synthesizer=SynthesizerConfig(),
        )
        prompt = render_synthesis_system_prompt(cfg)
        assert "Special values []" not in prompt

    def test_target_prompt_mixed_types(self):
        """Target prompt must handle mixed feature types."""
        from solon.config.schema import (
            DatasetConfig, SpecialValueSpec, FeatureSpec, DecisionSpec, TargetSpec
        )
        cfg = DatasetConfig(
            name="test",
            target=TargetSpec(column="label"),
            special_values=SpecialValueSpec(values=[], meanings={}, treatment="missing"),
            features=[
                FeatureSpec(name="loan_amnt", description="Loan amount", type="numeric"),
                FeatureSpec(name="emp_title", description="Job title", type="string"),
            ],
            decisions=DecisionSpec(),
        )
        prompt = render_target_system_prompt(cfg)
        assert "All features are numeric-valued" not in prompt

    def test_synthesis_prompt_mixed_types(self):
        """Synthesis prompt must not say 'all numeric' for mixed-type datasets."""
        from solon.config.schema import (
            DatasetConfig, SpecialValueSpec, FeatureSpec, SynthesizerConfig, TargetSpec
        )
        cfg = DatasetConfig(
            name="test",
            target=TargetSpec(column="label"),
            special_values=SpecialValueSpec(values=[], meanings={}, treatment="missing"),
            features=[
                FeatureSpec(name="loan_amnt", description="Loan amount", type="numeric"),
                FeatureSpec(name="emp_title", description="Job title", type="string"),
            ],
            synthesizer=SynthesizerConfig(),
        )
        prompt = render_synthesis_system_prompt(cfg)
        assert "all numeric" not in prompt
        assert "mixed types" in prompt

    def test_target_prompt_no_special_values(self):
        """Target prompt must not say 'Treat these as MISSING DATA' when no special values."""
        from solon.config.schema import (
            DatasetConfig, SpecialValueSpec, FeatureSpec, DecisionSpec, TargetSpec
        )
        cfg = DatasetConfig(
            name="test",
            target=TargetSpec(column="label"),
            special_values=SpecialValueSpec(values=[], meanings={}, treatment="missing"),
            features=[
                FeatureSpec(name="loan_amnt", description="Loan amount", type="numeric"),
            ],
            decisions=DecisionSpec(),
        )
        prompt = render_target_system_prompt(cfg)
        # Should NOT have the special value enumeration block
        assert "Treat these as MISSING DATA" not in prompt
        # Should have the alternative missing value guidance
        assert "missing" in prompt.lower() or "Missing" in prompt
