"""Shared preprocessing: special value handling, feature column helpers."""

import numpy as np
import pandas as pd
from solon.config.schema import DatasetConfig


def replace_special_values(df, config: DatasetConfig):
    """Replace special values with NaN for numeric operations (DataFrame or Series).

    Uses mask/isin rather than DataFrame.replace(list, nan), which raises
    IndexError on some frames under pandas 3.0.
    """
    return df.mask(df.isin(config.special_values.values))


def get_feature_columns(df: pd.DataFrame, config: DatasetConfig) -> list[str]:
    """Return feature column names (exclude target and cluster columns)."""
    exclude = {config.target.column, "cluster"}
    return [c for c in df.columns if c not in exclude]


def clean_and_impute(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    """Replace special values with NaN, impute numeric columns with median.

    Returns only numeric feature columns (suitable for scaling/clustering).
    """
    features = get_feature_columns(df, config)
    clean = df[features].copy()
    clean = replace_special_values(clean, config)
    numeric = clean.select_dtypes(include="number")
    medians = numeric.median()
    numeric = numeric.fillna(medians)
    return numeric
