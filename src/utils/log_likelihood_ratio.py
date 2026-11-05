import numpy as np

def calculate_substitution_score(
    weighted_counts: np.ndarray,
    position: int,
    wt: str,
    mut: str,
    alpha: float = 1.0,
) -> float:
    """"""
    site_counts = weighted_counts[position - 1]
    if site_counts.sum() == 0:
        return np.nan
    
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    wt_index = alphabet.index(wt)
    mut_index = alphabet.index(mut)

    wt_count = site_counts[wt_index]
    mut_count = site_counts[mut_index]
    pseudo_per_aa = alpha / 20
    score = -np.log((mut_count + pseudo_per_aa) / (wt_count + pseudo_per_aa))

    return float(score)
