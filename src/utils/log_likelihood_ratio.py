"""Scores amino acid substitutions from weighted alignment counts."""

import numpy as np


_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def calculate_substitution_score(
    weighted_counts: np.ndarray,
    position: int,
    wt: str,
    mut: str,
    alpha: float = 1.0,
) -> float:
    """Calculates the negative log ratio of mutant to wild-type counts.

    Args:
        weighted_counts: Nonnegative counts of shape (n_positions, 20), with
            columns ordered as ACDEFGHIKLMNPQRSTVWY.
        position: One-based residue position in the alignment.
        wt: Uppercase, single-letter code for the wild-type amino acid.
        mut: Uppercase, single-letter code for the mutant amino acid.
        alpha: Total pseudocount distributed equally among the 20 amino acids.
            Use a positive value to keep scores finite for unobserved residues.

    Returns:
        The negative natural logarithm of the mutant-to-wild-type count ratio
        after adding pseudocounts. Positive scores favor the wild-type residue.
        Returns NaN if the site has no observed amino acids.

    Raises:
        IndexError: If position - 1 is outside the array's index range.
        ValueError: If an amino acid code is absent from the alphabet and the
            site has observed amino acids.
    """
    site_counts = weighted_counts[position - 1]
    if site_counts.sum() == 0:
        return np.nan

    wt_index = _AMINO_ACIDS.index(wt)
    mut_index = _AMINO_ACIDS.index(mut)

    wt_count = site_counts[wt_index]
    mut_count = site_counts[mut_index]
    pseudocount_per_amino_acid = alpha / 20
    score = -np.log(
        (mut_count + pseudocount_per_amino_acid)
        / (wt_count + pseudocount_per_amino_acid)
    )

    return float(score)
