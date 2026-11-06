# Kyte–Doolittle hydropathy index
# AAindex: KYTJ820101
HYDROPHOBICITY = {
    "A":  1.8, "C":  2.5, "D": -3.5, "E": -3.5,
    "F":  2.8, "G": -0.4, "H": -3.2, "I":  4.5,
    "K": -3.9, "L":  3.8, "M":  1.9, "N": -3.5,
    "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8,
    "T": -0.7, "V":  4.2, "W": -0.9, "Y": -1.3,
}

# Krigbaum–Komoriya side-chain volume
# AAindex: KRIW790103；保留数据库原始量表值
SIDE_CHAIN_VOLUME = {
    "A":  27.5, "C":  44.6, "D":  40.0, "E":  62.0,
    "F": 115.5, "G":   0.0, "H":  79.0, "I":  93.5,
    "K": 100.0, "L":  93.5, "M":  94.1, "N":  58.7,
    "P":  41.9, "Q":  80.7, "R": 105.0, "S":  29.3,
    "T":  51.3, "V":  71.5, "W": 145.5, "Y": 117.3,
}

# Simplified side-chain charge: D/E = -1, K/R = +1, all others (including H) = 0.
CHARGE = {
    "A": 0.0, "C": 0.0, "D": -1.0, "E": -1.0,
    "F": 0.0, "G": 0.0, "H":  0.0, "I":  0.0,
    "K": 1.0, "L": 0.0, "M":  0.0, "N":  0.0,
    "P": 0.0, "Q": 0.0, "R":  1.0, "S":  0.0,
    "T": 0.0, "V": 0.0, "W":  0.0, "Y":  0.0,
}

# Feature convention: only F/W/Y count as aromatic; H is excluded.
AROMATICITY = {
    "A": 0.0, "C": 0.0, "D": 0.0, "E": 0.0,
    "F": 1.0, "G": 0.0, "H": 0.0, "I": 0.0,
    "K": 0.0, "L": 0.0, "M": 0.0, "N": 0.0,
    "P": 0.0, "Q": 0.0, "R": 0.0, "S": 0.0,
    "T": 0.0, "V": 0.0, "W": 1.0, "Y": 1.0,
}

PRO = {
    "A": 0.0, "C": 0.0, "D": 0.0, "E": 0.0,
    "F": 0.0, "G": 0.0, "H": 0.0, "I": 0.0,
    "K": 0.0, "L": 0.0, "M": 0.0, "N": 0.0,
    "P": 1.0, "Q": 0.0, "R": 0.0, "S": 0.0,
    "T": 0.0, "V": 0.0, "W": 0.0, "Y": 0.0,
}

GLY = {
    "A": 0.0, "C": 0.0, "D": 0.0, "E": 0.0,
    "F": 0.0, "G": 1.0, "H": 0.0, "I": 0.0,
    "K": 0.0, "L": 0.0, "M": 0.0, "N": 0.0,
    "P": 0.0, "Q": 0.0, "R": 0.0, "S": 0.0,
    "T": 0.0, "V": 0.0, "W": 0.0, "Y": 0.0,
}


class BiophysEncoder:
    """Adds mutant-minus-wild-type biophysical features to mutation records."""

    def __init__(self, path: str) -> None:
        self.mutations = []
        self.read_mutations(path)

    def read_mutations(self, path: str) -> None:
        """Replaces mutation records with entries read from a text file.

        Existing records and their features are preserved if reading or parsing
        fails. New records replace them only after the whole file is parsed.

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

        mutations = []
        for line in lines:
            _, mutation = line.split()
            wild_type = mutation[0]
            position = int(mutation[1:-1])
            mutant = mutation[-1]
            mutations.append({
                "mutation": mutation,
                "wt": wild_type,
                "position": position,
                "mut": mutant,
            })
        self.mutations = mutations

    def encode(self) -> None:
        """Adds P01-P06 in place, each using the signed difference mut - wt.

        P01 is Kyte-Doolittle hydropathy; P02 is Krigbaum-Komoriya side-chain
        volume; P03 is simplified charge; P04 is the F/W/Y aromatic indicator.
        P05 and P06 are Pro and Gly indicators, respectively: introduction is
        +1, removal is -1, and no change is 0. Inputs are assumed to contain
        standard uppercase amino-acid codes.
        """
        for record in self.mutations:
            mut = record["mut"]
            wt = record["wt"]
            record["P01"] = HYDROPHOBICITY[mut] - HYDROPHOBICITY[wt]
            record["P02"] = SIDE_CHAIN_VOLUME[mut] - SIDE_CHAIN_VOLUME[wt]
            record["P03"] = CHARGE[mut] - CHARGE[wt]
            record["P04"] = AROMATICITY[mut] - AROMATICITY[wt]
            record["P05"] = PRO[mut] - PRO[wt]
            record["P06"] = GLY[mut] - GLY[wt]
