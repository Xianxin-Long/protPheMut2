"""Split mutation data by reference site to keep sites within one partition."""

import numpy as np
import pandas as pd
from sklearn import model_selection


def split_by_position(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split rows into training and test sets without shared sites.

    Args:
        df: Data for one protein, containing a Site column.
        test_size: Fraction of distinct sites assigned to the test set.
        random_state: Seed controlling the grouped split.

    Returns:
        Training and test row positions for use with df.iloc, not index labels.
    """
    groups = df["Site"].to_numpy()
    splitter = model_selection.GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return train_idx, test_idx


def cross_validation_split(
    train_dataset: pd.DataFrame,
    n_splits: int = 3,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build stratified CV folds that keep each site's mutations together.

    Args:
        train_dataset: Training rows with label and Site columns.
        n_splits: Number of validation folds.
        random_state: Seed controlling the shuffled grouped folds.

    Returns:
        Fit/validation row positions relative to train_dataset. Use these with
        its feature matrix or .iloc, not with the original unsplit dataset.
    """
    splitter = model_selection.StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    return list(splitter.split(
        train_dataset,
        y=train_dataset["label"],
        groups=train_dataset["Site"],
    ))
