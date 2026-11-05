"""Parses alignments and encodes mutation-level evolutionary features.

A3M alignments use ColabFold chain-length metadata. Residues use alphabetical
amino-acid order, with separate codes for gaps and unknown residues.
Mutation positions are one-based, while alignment slices are zero-based.
"""

import dataclasses

import numpy as np

from src.utils import entropy
from src.utils import log_likelihood_ratio
from src.utils import mutual_information


_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def _remove_a3m_insertions(sequence: str) -> str:
    """Removes lowercase insertions and dots from an A3M sequence."""
    return "".join(
        character
        for character in sequence
        if not (character.islower() or character == ".")
    )


@dataclasses.dataclass
class EncodedMSA:
    """An encoded multiple sequence alignment and its chain boundaries.

    Attributes:
        encoded_msa: Array of shape (n_sequences, n_positions) with uint8
            residue codes. Standard amino acids use codes 0 through 19 in
            alphabetical order, gaps use 20, and unknown residues use 21.
        chain_slices: Zero-based, half-open (start, end) column boundaries
            for each chain, in alignment order.
    """

    encoded_msa: np.ndarray
    chain_slices: tuple[tuple[int, int], ...]


class MSAParser:
    """Reads a ColabFold A3M alignment and identifies its chain boundaries.

    Attributes:
        chain_lengths: Chain lengths from the A3M metadata, or None before
            metadata is read. Valid input must contain this metadata.
        headers: Sequence headers without their leading greater-than signs.
        aligned_sequences: Sequences with lowercase insertions and dots removed.
        chain_slices: Zero-based, half-open (start, end) column boundaries.
    """

    def __init__(self, path: str) -> None:
        """Loads an alignment and calculates its chain boundaries.

        Args:
            path: Path to an A3M file containing chain-length metadata.
        """
        self.chain_lengths = None
        self.headers = []
        self.aligned_sequences = []
        self.chain_slices = []
        self.read_a3m(path)
        self.calc_chain_slices()

    def read_a3m(self, path: str) -> None:
        """Replaces sequence records and chain lengths with data from an A3M.

        Chain metadata must have the form ``#lengths counts``, where lengths
        is a comma-separated list of integers. Call calc_chain_slices() after
        reloading to update the stored chain boundaries.

        Args:
            path: Path to an A3M file with FASTA-style sequence headers.

        Raises:
            OSError: If the alignment file cannot be read.
            ValueError: If chain metadata cannot be parsed.
        """
        with open(path) as alignment_file:
            lines = list(alignment_file)

        self.chain_lengths = None
        current_header = None
        sequence_parts = []
        records = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                chain_lengths, _ = line[1:].split()
                self.chain_lengths = [
                    int(length) for length in chain_lengths.split(",")
                ]
                continue
            if line.startswith(">"):
                if current_header is not None:
                    records.append((current_header, "".join(sequence_parts)))
                current_header = line[1:]
                sequence_parts = []
            else:
                sequence_parts.append(line)
        records.append((current_header, "".join(sequence_parts)))

        self.headers = [header for header, _ in records]
        self.aligned_sequences = [
            _remove_a3m_insertions(sequence) for _, sequence in records
        ]

    def calc_chain_slices(self) -> None:
        """Replaces chain slices using the chain lengths read from metadata."""
        self.chain_slices = []
        start = 0
        for length in self.chain_lengths:
            end = start + length
            self.chain_slices.append((start, end))
            start = end


class MSAEncoder:
    """Preprocesses aligned sequences and encodes their residues as integers.

    Attributes:
        headers: Sequence headers in the same order as aligned_sequences.
        aligned_sequences: Aligned sequences with A3M insertions removed.
        chain_slices: Zero-based, half-open (start, end) column boundaries.
    """

    def __init__(
        self,
        headers: list[str],
        aligned_sequences: list[str],
        chain_slices: list[tuple[int, int]],
    ) -> None:
        """Stores sequence records and their chain boundaries for preprocessing.

        Args:
            headers: One header for each aligned sequence.
            aligned_sequences: Equal-length sequences in header order.
            chain_slices: Zero-based, half-open boundaries for each chain.
        """
        self.headers = headers
        self.aligned_sequences = aligned_sequences
        self.chain_slices = chain_slices

    def select_chain(self, chain_index: int) -> None:
        """Keeps one chain and resets its column boundaries to start at zero.

        Args:
            chain_index: Index into chain_slices, using Python list indexing.

        Raises:
            IndexError: If chain_index is outside the available chain slices.
        """
        start, end = self.chain_slices[chain_index]
        self.aligned_sequences = [
            sequence[start:end] for sequence in self.aligned_sequences
        ]
        self.chain_slices = [(0, end - start)]

    def deduplicate_msa(self) -> None:
        """Keeps the first occurrence and header of each distinct sequence."""
        seen = set()
        unique_headers = []
        unique_sequences = []

        for header, sequence in zip(self.headers, self.aligned_sequences):
            if sequence in seen:
                continue
            seen.add(sequence)
            unique_headers.append(header)
            unique_sequences.append(sequence)

        self.headers = unique_headers
        self.aligned_sequences = unique_sequences

    def filter_msa_by_coverage(self, min_coverage: float = 0.5) -> None:
        """Keeps sequences meeting the residue coverage threshold in each chain.

        Only the 20 standard amino acids count toward coverage. Each chain
        slice must have positive length. Headers retain their sequence order.

        Args:
            min_coverage: Minimum fraction of standard amino acids per chain.
        """
        kept_headers = []
        kept_sequences = []
        for header, sequence in zip(self.headers, self.aligned_sequences):
            coverages = []
            for start, end in self.chain_slices:
                segment = sequence[start:end]
                n_valid = sum(residue in _AMINO_ACIDS for residue in segment)
                coverage = n_valid / (end - start)
                coverages.append(coverage)
            if all(coverage >= min_coverage for coverage in coverages):
                kept_headers.append(header)
                kept_sequences.append(sequence)

        self.headers = kept_headers
        self.aligned_sequences = kept_sequences

    def encode_msa(self) -> EncodedMSA:
        """Returns a new encoding of the current, preprocessed alignment.

        Returns:
            An EncodedMSA with uint8 residue codes and the current chain
            boundaries. Standard amino acids use codes 0 through 19 in
            alphabetical order, gaps use 20, and X uses 21.

        Raises:
            KeyError: If a sequence contains an unsupported residue symbol.
            ValueError: If the aligned sequences have different lengths.
        """
        residue_to_code = {
            residue: index for index, residue in enumerate(_AMINO_ACIDS)
        }
        residue_to_code["-"] = 20
        residue_to_code["X"] = 21

        rows = []
        for sequence in self.aligned_sequences:
            row = []
            for residue in sequence:
                row.append(residue_to_code[residue])
            rows.append(row)

        return EncodedMSA(
            encoded_msa=np.array(rows, dtype=np.uint8),
            chain_slices=tuple(self.chain_slices),
        )


class EvolEncoder:
    """Adds alignment-derived evolutionary features to mutation records.

    Attributes:
        mutations: Records containing mutation notation, wild-type residue
            (wt), one-based position, and mutant residue (mut). Calling encode()
            adds the numeric F01, F02, and F03 features to these records.
        msa: Encoded alignment with the mutated chain first.
        is_complex: Whether to include cross-chain mutual information as F03.
    """

    def __init__(
        self,
        path: str,
        msa: EncodedMSA,
        is_complex: bool,
    ) -> None:
        """Loads mutation records and stores their alignment context.

        Args:
            path: Path to the mutation file, with two whitespace-separated
                fields per line. The second field uses notation such as A42V.
            msa: Encoded alignment with mutations mapped to its first chain.
            is_complex: Whether F03 should contain cross-chain MI scores.
        """
        self.mutations = []
        self.read_mutations(path)
        self.msa = msa
        self.is_complex = is_complex

    def read_mutations(self, path: str) -> None:
        """Replaces mutation records with entries read from a text file.

        Args:
            path: Path to a file with an identifier and a mutation such as
                A42V on each line. Residue positions are one-based.

        Raises:
            OSError: If the mutation file cannot be read.
            ValueError: If a line does not have two fields or its mutation
                position cannot be parsed as an integer.
        """
        with open(path) as mutation_file:
            lines = list(mutation_file)

        self.mutations = []
        for line in lines:
            _, mutation = line.split()
            wild_type = mutation[0]
            position = int(mutation[1:-1])
            mutant = mutation[-1]
            self.mutations.append({
                "mutation": mutation,
                "wt": wild_type,
                "position": position,
                "mut": mutant,
            })

    def encode(self) -> None:
        """Adds F01, F02, and F03 to each stored mutation record in place.

        F01 is sequence-weighted site entropy normalized by log(20). F02 is
        the negative natural-log mutant-to-wild-type count ratio with a
        uniform pseudocount. F03 is the mean of the three highest eligible
        APC-corrected cross-chain MI scores, or NaN when unavailable or when
        is_complex is False. Mutation positions must index the first chain.
        """
        a_start, a_end = self.msa.chain_slices[0]
        length_a = a_end - a_start

        site_entropies, counts = entropy.calculate_site_entropy(
            self.msa.encoded_msa
        )
        mi_scores = mutual_information.calculate_cross_chain_mi(
            encoded_msa=self.msa.encoded_msa,
            chain_length_a=length_a,
        )

        for record in self.mutations:
            record["F01"] = float(site_entropies[record["position"] - 1])
            record["F02"] = log_likelihood_ratio.calculate_substitution_score(
                weighted_counts=counts,
                position=record["position"],
                wt=record["wt"],
                mut=record["mut"],
            )
            if self.is_complex:
                column_index = record["position"] - 1
                record["F03"] = float(mi_scores.get(column_index, np.nan))
            else:
                record["F03"] = float("nan")
