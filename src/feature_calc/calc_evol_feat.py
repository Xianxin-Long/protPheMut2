def read_query_fasta(path: str) -> dict:
    """"""
    with open(path) as f:
        lines = f.readlines()

    sequence_parts = []
    for line in lines:
        line = line.strip()
        if line.startswith(">"):
            continue
        if not line:
            continue
        sequence_parts.append(line)

    sequence_text = "".join(sequence_parts)
    chain_sequences = sequence_text.split(":")
    query_sequence = "".join(chain_sequences)

    chain_lengths = []
    chain_slices = []
    start = 0
    for chain in chain_sequences:
        end = start + len(chain)
        chain_lengths.append(len(chain))
        chain_slices.append((start, end))
        start = end

    return {
        "chain_sequences": chain_sequences,
        "chain_lengths": chain_lengths,
        "chain_slices": chain_slices,
        "query_sequence": query_sequence,
    }


def remove_a3m_insertions(sequence: str) -> str:
    """"""
    removed = ""

    for ch in sequence:
        if not (ch.islower() or ch == "."):
            removed += ch

    return removed


def read_a3m(path: str) -> dict:
    """"""
    with open(path) as f:
        lines = f.readlines()

    chain_lengths, chain_copies = None, None
    current_header = None
    sequence_parts = []
    records = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        elif line.startswith("#"):
            chain_lengths, chain_copies = line[1:].split()
            chain_lengths = list(map(int, chain_lengths.split(",")))
            chain_copies = list(map(int, chain_copies.split(",")))
            continue
        elif line[0] == ">":
            if current_header != None:
                records.append((current_header, "".join(sequence_parts)))
            current_header = line[1:]
            sequence_parts = []
        else:
            sequence_parts.append(line)
    records.append((current_header, "".join(sequence_parts)))

    headers = [records[i][0] for i in range(len(records))]
    aligned_sequences = [remove_a3m_insertions(records[i][1]) for i in range(len(records))]

    return {
        "chain_lengths": chain_lengths,
        "chain_copies": chain_copies,
        "headers": headers,
        "aligned_sequences": aligned_sequences
    }


def deduplicate_msa(msa: dict) -> dict:
    """"""
    seen = set()
    unique_header = []
    unique_sequence = []

    for header, sequence in zip(msa["headers"], msa["aligned_sequences"]):
        if sequence in seen:
            continue
        else:
            seen.add(sequence)
            unique_header.append(header)
            unique_sequence.append(sequence)

    msa["headers"], msa["aligned_sequences"] = unique_header, unique_sequence

    return msa

msa = {"headers": ["query", "hit1", "hit2", "hit3"], "aligned_sequences": ["AC-D", "AC-D", "AT-D", "AC-D"]}
print(deduplicate_msa(msa))
