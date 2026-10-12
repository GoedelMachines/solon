"""Load dataset configuration from YAML and merge with CLI overrides."""

import yaml
from .schema import (
    DatasetConfig, RunConfig, FeatureSpec, TargetSpec, DecisionSpec,
    SpecialValueSpec, EvaluationConfig, SynthesizerConfig, PipelineDefaults,
)


def _build_features(raw_list: list[dict]) -> list[FeatureSpec]:
    return [FeatureSpec(**f) for f in raw_list]


def load_dataset_config(yaml_path: str) -> DatasetConfig:
    """Load a DatasetConfig from a YAML file."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    target = TargetSpec(**raw.get("target", {}))
    decisions = DecisionSpec(**raw.get("decisions", {}))

    sv_raw = raw.get("special_values", {})
    # Convert string keys in meanings to int
    meanings = {int(k): v for k, v in sv_raw.get("meanings", {}).items()}
    special_values = SpecialValueSpec(
        values=sv_raw.get("values", [-7, -8, -9]),
        meanings=meanings,
        treatment=sv_raw.get("treatment", "missing"),
    )

    features = _build_features(raw.get("features", []))
    summary_features = raw.get("summary_features", [])

    eval_raw = raw.get("evaluation", {})
    _eval_defaults = EvaluationConfig()
    evaluation = EvaluationConfig(
        prohibited_features=eval_raw.get("prohibited_features", []),
        require_reason_codes=eval_raw.get("require_reason_codes", True),
        max_crash_rate=eval_raw.get("max_crash_rate", 0.1),
        weights=eval_raw.get("weights", _eval_defaults.weights),
        simplicity_thresholds=eval_raw.get("simplicity_thresholds", _eval_defaults.simplicity_thresholds),
    )

    synth_raw = raw.get("synthesizer", {})
    synthesizer = SynthesizerConfig(**synth_raw) if synth_raw else SynthesizerConfig()

    defaults_raw = raw.get("defaults", {})
    defaults = PipelineDefaults(**defaults_raw) if defaults_raw else PipelineDefaults()

    return DatasetConfig(
        name=raw.get("name", "unnamed"),
        description=raw.get("description", ""),
        target=target,
        decisions=decisions,
        special_values=special_values,
        features=features,
        summary_features=summary_features,
        regulatory_context=raw.get("regulatory_context", ""),
        data_observations=raw.get("data_observations", ""),
        decision_framework=raw.get("decision_framework", ""),
        product_context=raw.get("product_context", ""),
        evaluation=evaluation,
        synthesizer=synthesizer,
        defaults=defaults,
    )


def build_run_config(yaml_path: str, **cli_overrides) -> RunConfig:
    """Load dataset config and merge with CLI overrides to produce a RunConfig."""
    dataset = load_dataset_config(yaml_path)

    # Extract RunConfig fields from overrides, pass the rest through
    run_kwargs = {"dataset": dataset}
    for key in ("cases_csv", "full_dataset_csv", "output_dir", "provider", "model",
                 "n_candidates", "n_generations", "top_k", "targets_path",
                 "batch_size", "holdout_size"):
        if key in cli_overrides and cli_overrides[key] is not None:
            run_kwargs[key] = cli_overrides[key]

    return RunConfig(**run_kwargs)
