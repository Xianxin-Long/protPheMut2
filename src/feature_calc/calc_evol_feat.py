import numpy as np
from numba import njit

class MSAparser:
    """"""
    def __init__(self, path: str) -> None:
        """"""
        self.read_a3m(path)
        self.calc_chain_slices()

    def read_a3m(self, path: str) -> None:
        """"""
        def remove_a3m_insertions(sequence: str) -> str:
            """"""
            removed = ""

            for ch in sequence:
                if not (ch.islower() or ch == "."):
                    removed += ch

            return removed
        
        with open(path) as f:
            lines = f.readlines()

        self.chain_lengths = None
        current_header = None
        sequence_parts = []
        records = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            elif line.startswith("#"):
                self.chain_lengths, _ = line[1:].split()
                self.chain_lengths = list(map(int, self.chain_lengths.split(",")))
                continue
            elif line[0] == ">":
                if current_header != None:
                    records.append((current_header, "".join(sequence_parts)))
                current_header = line[1:]
                sequence_parts = []
            else:
                sequence_parts.append(line)
        records.append((current_header, "".join(sequence_parts)))

        self.headers = [records[i][0] for i in range(len(records))]
        self.aligned_sequences = [remove_a3m_insertions(records[i][1]) for i in range(len(records))]

    def calc_chain_slices(self) -> None:
        """"""
        self.chain_slices = []
        start = 0
        for length in self.chain_lengths:
            end = start + length
            self.chain_slices.append((start, end))
            start = end


class MSAencoder:
    """"""
    def __init__(self, headers: list[str], aligned_sequences: list[str], chain_slices: list[tuple[int, int]]) -> None:
        """"""
        self.headers = headers
        self.aligned_sequences = aligned_sequences
        self.chain_slices = chain_slices

    def select_chain(self, chain_index: int) -> None:
        start, end = self.chain_slices[chain_index]

        self.aligned_sequences = [
            sequence[start:end]
            for sequence in self.aligned_sequences
        ]

        self.chain_slices = [(0, end - start)]

    def deduplicate_msa(self) -> None:
        """"""
        seen = set()
        unique_header = []
        unique_sequence = []

        for header, sequence in zip(self.headers, self.aligned_sequences):
            if sequence in seen:
                continue
            else:
                seen.add(sequence)
                unique_header.append(header)
                unique_sequence.append(sequence)

        self.headers, self.aligned_sequences = unique_header, unique_sequence

    def filter_msa_by_coverage(self, min_coverage: float = 0.5) -> None:
        """"""
        AA20 = "ACDEFGHIKLMNPQRSTVWY"

        kept_headers = []
        kept_sequences = []      
        for header, sequence in zip(self.headers, self.aligned_sequences):
            coverages = []
            for start, end in self.chain_slices:
                segment = sequence[start:end]
                n_valid = sum(ch in AA20 for ch in segment)
                coverage = n_valid / (end - start)
                coverages.append(coverage)
            if all(coverage >= min_coverage for coverage in coverages):
                kept_headers.append(header)
                kept_sequences.append(sequence)
        
        self.headers = kept_headers
        self.aligned_sequences = kept_sequences

    def encode_msa(self) -> None:
        """"""
        AA20 = "ACDEFGHIKLMNPQRSTVWY"
        aa_to_int = {aa: i for i, aa in enumerate(AA20)}
        aa_to_int["-"] = 20
        aa_to_int["X"] = 21

        rows = []
        for sequence in self.aligned_sequences:
            row = []
            for ch in sequence:
                row.append(aa_to_int[ch])
            rows.append(row)

        self.encoded_msa = np.array(rows, dtype=np.uint8)


@njit
def pair_identity(
    seq_a: np.ndarray, 
    seq_b: np.ndarray
) -> float:
    """"""
    valid_a = seq_a < 20
    valid_b = seq_b < 20

    compared = valid_a | valid_b
    matched = valid_a & valid_b & (seq_a == seq_b)

    n_compared = np.count_nonzero(compared)
    n_matched = np.count_nonzero(matched)

    return n_matched / n_compared
    

@njit
def calculate_site_entropy(
    encoded_msa: np.ndarray,
    identity_threshold: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """"""
    #
    n_sequences = encoded_msa.shape[0]
    neighbor_counts = np.ones(n_sequences, dtype=np.int64)

    for i in range(n_sequences):
        for j in range(i + 1, n_sequences):
            identity= pair_identity(encoded_msa[i], encoded_msa[j])
            if identity >= identity_threshold:
                neighbor_counts[i] += 1
                neighbor_counts[j] += 1

    weights = 1.0 / neighbor_counts

    #
    n_sequences, n_positions = encoded_msa.shape

    counts = np.zeros((n_positions, 20), dtype=np.float64,)
    for i in range(n_sequences):
        for j in range(n_positions):
            aa = encoded_msa[i, j]
            if aa < 20:
                counts[j, aa] += weights[i]

    frequencies = np.full((n_positions, 20), np.nan, dtype=np.float64)
    for j in range(n_positions):
        total_weight = counts[j].sum()
        if total_weight > 0:
            frequencies[j] = counts[j] / total_weight

    #
    n_positions = frequencies.shape[0]
    entropies = np.full(n_positions, np.nan, dtype=np.float64)

    for j in range(n_positions):
        p = frequencies[j]
        if np.isnan(p).any():
            continue
        p_positive = p[p > 0]
        entropy = -np.sum(p_positive * np.log(p_positive))
        entropies[j] = entropy / np.log(20)

    return entropies, counts

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

class mutationEncoder:
    def __init__(self, path, encoded_msa):
        self.read_mutations(path)
        self.encoded_msa = encoded_msa

    def read_mutations(self, path: str) -> None:
        with open(path) as f:
            lines = f.readlines()

        self.mutations = []
        for line in lines:
            line = line.strip()
            _, mutation = line.split()
            wt = mutation[0]
            position = int(mutation[1:-1])
            mut = mutation[-1]
            self.mutations.append({
                "mutation": mutation,
                "wt": wt,
                "position": position,
                "mut": mut 
            })

    def encode_conservation(self) -> None:
        site_entropies, counts = calculate_site_entropy(self.encoded_msa)

        for record in self.mutations:
            record["F01"] = float(
                site_entropies[record["position"] - 1]
            )
            record["F02"] = calculate_substitution_score(
                weighted_counts=counts,
                position=record["position"],
                wt=record["wt"],
                mut=record["mut"],
            )




