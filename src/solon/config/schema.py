"""Dataclass definitions for dataset and run configuration."""

from dataclasses import dataclass, field


@dataclass
class FeatureSpec:
    name: str
    description: str
    type: str = "integer"
    range: str = ""
    direction: str = ""  # "higher_better", "lower_better", etc.
    distribution: str = ""
    missing_pct: float = 0.0


@dataclass
class TargetSpec:
    column: str
    positive_label: int | str = 0  # "good" outcome
    negative_label: int | str = 1  # "bad" outcome
    positive_meaning: str = "good outcome"
    negative_meaning: str = "bad outcome"


@dataclass
class DecisionSpec:
    approve: str = "APPROVE"
    deny: str = "DENY"
    approve_maps_to: str = "positive"  # "positive" or "negative" label
    deny_maps_to: str = "negative"


@dataclass
class SpecialValueSpec:
    values: list[int] = field(default_factory=lambda: [-7, -8, -9])
    meanings: dict[int, str] = field(default_factory=dict)
    treatment: str = "missing"  # "missing", "zero", "median"


@dataclass
class EvaluationConfig:
    prohibited_features: list[str] = field(default_factory=list)
    require_reason_codes: bool = True
    max_crash_rate: float = 0.1
    weights: dict[str, float] = field(default_factory=lambda: {
        "accuracy": 0.4,
        "simplicity": 0.2,
        "robustness": 0.2,
        "coverage": 0.2,
    })
    simplicity_thresholds: dict[str, int] = field(default_factory=lambda: {
        "ideal_max": 10,
        "good_max": 20,
        "acceptable_max": 40,
    })


@dataclass
class SynthesizerConfig:
    function_name: str = "underwrite"
    return_fields: list[str] = field(default_factory=lambda: ["decision", "reason_codes", "score"])


@dataclass
class PipelineDefaults:
    sample_size: int = 100
    n_clusters: int = 20
    holdout_size: int = 500
    batch_size: int = 10
    n_candidates: int = 5
    n_generations: int = 10
    top_k: int = 3


@dataclass
class DatasetConfig:
    name: str
    description: str = ""
    target: TargetSpec = field(default_factory=TargetSpec)
    decisions: DecisionSpec = field(default_factory=DecisionSpec)
    special_values: SpecialValueSpec = field(default_factory=SpecialValueSpec)
    features: list[FeatureSpec] = field(default_factory=list)
    summary_features: list[str] = field(default_factory=list)
    regulatory_context: str = ""
    data_observations: str = ""
    decision_framework: str = ""
    product_context: str = ""
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    synthesizer: SynthesizerConfig = field(default_factory=SynthesizerConfig)
    defaults: PipelineDefaults = field(default_factory=PipelineDefaults)

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    @property
    def special_value_list(self) -> list[int]:
        return self.special_values.values


@dataclass
class RunConfig:
    dataset: DatasetConfig
    cases_csv: str
    full_dataset_csv: str | None = None
    output_dir: str = "outputs"
    provider: str = "anthropic"
    model: str | None = None
    n_candidates: int | None = None
    n_generations: int | None = None
    top_k: int | None = None
    targets_path: str | None = None
    batch_size: int | None = None
    holdout_size: int | None = None

    def __post_init__(self):
        """Fill None fields from dataset defaults."""
        d = self.dataset.defaults
        if self.n_candidates is None:
            self.n_candidates = d.n_candidates
        if self.n_generations is None:
            self.n_generations = d.n_generations
        if self.top_k is None:
            self.top_k = d.top_k
        if self.batch_size is None:
            self.batch_size = d.batch_size
        if self.holdout_size is None:
            self.holdout_size = d.holdout_size
