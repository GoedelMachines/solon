"""Jinja2 templates for system prompts, built from DatasetConfig."""

from jinja2 import Template
from solon.config.schema import DatasetConfig

# ---------------------------------------------------------------------------
# Target generation system prompt
# ---------------------------------------------------------------------------

TARGET_SYSTEM_TEMPLATE = Template("""\
You are an expert analyst evaluating {{ product_context | trim }}

REGULATORY CONTEXT:

{{ regulatory_context }}
FEATURE DEFINITIONS AND DATA DISTRIBUTIONS:
{% if mixed_types %}
Features are a mix of numeric, categorical, and string types (noted in each definition).
{% else %}
All features are {{ feature_type }}-valued.
{% endif %}
The distributions below are from the actual applicant pool.

{% if special_values.values %}
{% for sv in special_values.values -%}
Special value {{ sv }} = "{{ special_values.meanings.get(sv, 'special') }}"{{ ", " if not loop.last else "." }}
{%- endfor %}
 Treat these as MISSING DATA, not as numeric values.
{% else %}
Missing or unavailable values appear as blank/empty. When a value is missing, it may carry meaning (e.g., "months since last delinquency" being missing means no delinquency on record).
{% endif %}

{% for i, feat in features %}
{{ i }}. {{ feat.name }} — {{ feat.description }}
   {{ feat.distribution }}
   {{ "%.1f" | format(feat.missing_pct) }}% missing.
{% endfor %}

DATA OBSERVATIONS:
{{ data_observations }}
DECISION FRAMEWORK:

{{ decision_framework }}
{{ product_reminder }}

Output your decision as JSON with these fields:
- decision: "{{ decisions.approve }}" or "{{ decisions.deny }}"
- confidence: float 0.0-1.0
- reason_codes: list of 1-4 specific adverse action reason codes (even for approvals, note key risk factors)
- reasoning: 2-3 sentence explanation covering which dimensions drove the decision and any interaction effects considered\
""")


def render_target_system_prompt(config: DatasetConfig) -> str:
    """Render the target generation system prompt from a dataset config."""
    features_enum = list(enumerate(config.features, 1))
    feature_type = "integer"  # could be derived from config.features
    if config.features and config.features[0].type:
        feature_type = config.features[0].type

    types = set(f.type for f in config.features if f.type)
    mixed_types = len(types) > 1

    product_reminder = ""
    if config.product_context:
        product_reminder = f"Remember: {config.product_context.strip()}"

    return TARGET_SYSTEM_TEMPLATE.render(
        product_context=config.product_context or config.description,
        regulatory_context=config.regulatory_context,
        feature_type=feature_type,
        mixed_types=mixed_types,
        special_values=config.special_values,
        features=features_enum,
        data_observations=config.data_observations,
        decision_framework=config.decision_framework,
        product_reminder=product_reminder,
        decisions=config.decisions,
    )


# ---------------------------------------------------------------------------
# Program synthesis system prompt
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM_TEMPLATE = Template("""\
You are an expert at writing deterministic decision rules as Python code.

TASK: Write a Python function that evaluates applications.
The function must be deterministic — same input always produces same output. No randomness, no ML models.

FUNCTION SIGNATURE (do not change):
```python
def {{ function_name }}(application: dict) -> dict:
```

{% if mixed_types %}
INPUT: A dict with these keys (mixed types — numeric, categorical, and string):
{%- else %}
INPUT: A dict with these keys (all {{ feature_type }} values):
{%- endif %}
{% for feat in features -%}
- {{ feat.name }} ({{ feat.description | truncate(60) }})
{% endfor %}
{% if special_values.values %}
Special values {{ special_values.values }} mean missing/unavailable data. Use the get() helper.
{% else %}
Missing values appear as None or NaN. Use the get() helper to safely access features with a default.
{% endif %}

OUTPUT: dict with:
{% for field in return_fields -%}
- {{ field }}{% if field == "decision" %}: "{{ decisions.approve }}" or "{{ decisions.deny }}"{% endif %}
{% if field == "reason_codes" %}: list of 1-4 strings explaining the decision{% endif %}
{% if field == "score" %}: float (lower = less risky){% endif %}
{% endfor %}

REQUIREMENTS:
1. Rules must be explicit conditional logic (if/elif/else). No ML, no math beyond arithmetic.
2. Every {{ decisions.deny }} must have at least one reason code.
3. Handle missing values (special values) gracefully — don't crash.
4. Code must be readable by a non-programmer compliance officer.
5. Use descriptive variable names and comments explaining the business logic.
6. Return the COMPLETE function body — everything between the helpers and the return.\
""")


def render_synthesis_system_prompt(config: DatasetConfig) -> str:
    """Render the synthesis system prompt from a dataset config."""
    feature_type = "integer"
    if config.features and config.features[0].type:
        feature_type = config.features[0].type

    types = set(f.type for f in config.features if f.type)
    mixed_types = len(types) > 1

    return SYNTHESIS_SYSTEM_TEMPLATE.render(
        function_name=config.synthesizer.function_name,
        feature_type=feature_type,
        mixed_types=mixed_types,
        features=config.features,
        special_values=config.special_values,
        return_fields=config.synthesizer.return_fields,
        decisions=config.decisions,
    )


# ---------------------------------------------------------------------------
# Program skeleton
# ---------------------------------------------------------------------------

SKELETON_TEMPLATE = Template("""\
import numpy as np


def {{ function_name }}(application: dict) -> dict:
    \"\"\"
    Evaluate an application and return a decision.

    Args:
        application: dict with keys matching feature names.
{% if special_values.values %}
            Special values: {{ special_values.values | join(', ') }} mean missing data.
{% else %}
            Missing values appear as None or NaN.
{% endif %}

    Returns:
        dict with:
{%- for field in return_fields %}
            - {{ field }}
{%- endfor %}
    \"\"\"
    # === Helper: treat special values as missing ===
    def get(feature, default=None):
        val = application.get(feature)
{% if special_values.values %}
        if val in ({{ special_values.values | join(', ') }},) or val is None:
{% else %}
        if val is None or (isinstance(val, float) and val != val):
{% endif %}
            return default
        return val

    # === YOUR RULES GO HERE ===
    # {RULES}
    # === END RULES ===
""")


def render_program_skeleton(config: DatasetConfig) -> str:
    """Render the program skeleton template from a dataset config."""
    return SKELETON_TEMPLATE.render(
        function_name=config.synthesizer.function_name,
        special_values=config.special_values,
        return_fields=config.synthesizer.return_fields,
    )
