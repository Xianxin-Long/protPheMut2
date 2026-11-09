"""Maps the first FASTA sequence to chain A of the first PDB model.

The output CSV contains one row per reference residue, with its one-based
position, PDB residue number, and insertion code. Unmapped reference positions
have an empty PDB residue number and insertion code.

Example:
    python scripts/residue_mapping.py reference.fasta structure.pdb mapping.csv
"""

import argparse

from Bio import Align
from Bio import PDB
from Bio import SeqIO
import pandas as pd


AMINO_ACIDS_3TO1 = {
    "CYS": "C", "ASP": "D", "SER": "S", "GLN": "Q", "LYS": "K",
    "ILE": "I", "PRO": "P", "THR": "T", "PHE": "F", "ASN": "N",
    "GLY": "G", "HIS": "H", "LEU": "L", "ARG": "R", "TRP": "W",
    "ALA": "A", "VAL": "V", "GLU": "E", "TYR": "Y", "MET": "M",
}


def read_fasta_1(fasta_file: str) -> str:
    """Reads the sequence from the first FASTA record.

    Args:
        fasta_file: Path to the input FASTA file. Later records are ignored.

    Returns:
        The first record's amino acid sequence as a string.

    Raises:
        OSError: If the input file cannot be read.
        StopIteration: If the file contains no FASTA records.
    """
    with open(fasta_file, encoding="utf-8") as fasta_handle:
        return str(next(SeqIO.parse(fasta_handle, "fasta")).seq)


def extract_chain_a_seq(
    pdb_file: str,
) -> tuple[str, list[tuple[str, int, str]]]:
    """Extracts canonical residues from chain A of the first PDB model.

    Args:
        pdb_file: Path to the input PDB file.

    Returns:
        A tuple containing the one-letter sequence and its ordered PDB residue
        IDs. Each ID contains the hetero flag, residue number, and insertion
        code. Residues absent from AMINO_ACIDS_3TO1 are skipped.

    Raises:
        OSError: If the input file cannot be read.
        KeyError: If the structure has no first model or no chain A.
        PDB.PDBExceptions.PDBConstructionException: If PDB parsing fails.
    """
    parser = PDB.PDBParser(PERMISSIVE=True)
    structure = parser.get_structure(None, pdb_file)
    model = structure[0]
    chain = model["A"]
    sequence = ""
    residue_ids = []
    for residue in chain.get_residues():
        amino_acid = AMINO_ACIDS_3TO1.get(residue.get_resname())
        if amino_acid is None:
            continue
        sequence += amino_acid
        residue_ids.append(residue.id)
    return sequence, residue_ids


def semi_global_alignment(seq1: str, seq2: str) -> Align.Alignment:
    """Aligns a structure sequence against a reference sequence.

    Args:
        seq1: Reference sequence, used as the alignment target.
        seq2: Structure sequence, used as the alignment query.

    Returns:
        The first optimal alignment, allowing missing terminal reference
        residues without a gap penalty.

    Raises:
        ValueError: If either sequence is empty.
    """
    aligner = Align.PairwiseAligner(
        match_score=2.0,
        mismatch_score=-3.0,
        open_gap_score=-5.0,
        extend_gap_score=-0.5,
        # Structures may omit residues at either end of the reference.
        end_deletion_score=0.0,
        mode="global",
    )
    return aligner.align(seq1, seq2)[0]


def build_mapping_table(
    alignment: Align.Alignment,
    residue_ids: list[tuple[str, int, str]],
) -> pd.DataFrame:
    """Builds a reference-to-structure residue mapping table.

    Args:
        alignment: Pairwise alignment with the reference as the target and the
            structure sequence as the query.
        residue_ids: PDB residue IDs in query sequence order. Each ID contains
            the hetero flag, residue number, and insertion code.

    Returns:
        A DataFrame with columns ref_pos (one-based reference position),
        struct_resnum (nullable PDB residue number), and insertion_code
        (stripped PDB insertion code). Every reference position has a row;
        unmapped positions have a missing number and an empty insertion code.
        Aligned mismatches are mapped as well as matches.
    """
    rows = [
        [position, None, ""]
        for position in range(1, len(alignment.target) + 1)
    ]
    for reference_index, structure_index in alignment.indices.T:
        # Biopython uses -1 for gaps, which have no residue correspondence.
        if reference_index == -1 or structure_index == -1:
            continue
        _, residue_number, insertion_code = residue_ids[int(structure_index)]
        rows[int(reference_index)][1] = residue_number
        rows[int(reference_index)][2] = insertion_code.strip()

    table = pd.DataFrame(
        rows, columns=["ref_pos", "struct_resnum", "insertion_code"],
    )
    # Nullable integers preserve residue numbers when some positions are absent.
    table["struct_resnum"] = table["struct_resnum"].astype("Int64")

    return table


def main() -> None:
    """Parses command-line paths and writes the residue mapping to CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fasta_file", help="Path to the FASTA file.", type=str)
    parser.add_argument("pdb_file", help="Path to the PDB file.", type=str)
    parser.add_argument(
        "output_file", help="Path to the output CSV file.", type=str,
    )

    args = parser.parse_args()
    reference_sequence = read_fasta_1(args.fasta_file)
    structure_sequence, residue_ids = extract_chain_a_seq(args.pdb_file)
    alignment = semi_global_alignment(reference_sequence, structure_sequence)
    mapping_table = build_mapping_table(alignment, residue_ids)

    mapping_table.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
