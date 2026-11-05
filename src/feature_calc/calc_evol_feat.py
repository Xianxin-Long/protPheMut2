import numpy as np
from src.utils.entropy import calculate_site_entropy
from src.utils.log_likelihood_ratio import calculate_substitution_score
from src.utils.mutual_information import calculate_cross_chain_mi

class MSAParser:
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


class MSAEncoder:
    """"""
    def __init__(
        self, 
        headers: list[str], 
        aligned_sequences: list[str], 
        chain_slices: list[tuple[int, int]]
    ) -> None:
        """"""
        self.headers = headers
        self.aligned_sequences = aligned_sequences
        self.chain_slices = chain_slices

    def select_chain(self, chain_index: int) -> None:
        start, end = self.chain_slices[chain_index]
        self.aligned_sequences = [sequence[start:end] for sequence in self.aligned_sequences]
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


class EvolEncoder:
    """"""
    def __init__(
        self,
        path: str,
        encoded_msa: np.ndarray,
        chain_slices: list[tuple[int, int]],
        is_complex: bool
    ) -> None:
        """"""
        self.read_mutations(path)
        self.encoded_msa = encoded_msa
        self.chain_slices = chain_slices
        self.is_complex = is_complex

    def read_mutations(self, path: str) -> None:
        """"""
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

    def encode(self) -> None:
        """"""
        a_start, a_end = self.chain_slices[0]
        length_a = a_end - a_start

        site_entropies, counts = calculate_site_entropy(self.encoded_msa)
        mi_scores = calculate_cross_chain_mi(
            encoded_msa=self.encoded_msa,
            chain_length_a=length_a,
        )

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
            if self.is_complex:
                column_index = record["position"] - 1
                record["F03"] = float(mi_scores.get(column_index, np.nan))
            else:
                record["F03"] = float('nan')

