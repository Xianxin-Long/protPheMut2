"""Local ColabFold CLI and Python API; exports paired/unpaired A3M and best PDB.

Command-line examples (from the project root)::

    python scripts/colabfold_runner.py data/7MYN/7MYN.fasta results/7MYN
    python scripts/colabfold_runner.py data/7MYN/7MYN.fasta results/7MYN --msa-only
    python scripts/colabfold_runner.py --help

Python API::

    from scripts.colabfold_runner import run_colabfold
    files = run_colabfold("data/SHP2/SHP2.fasta", "results/SHP2", True, name="SHP2_wt")

Prediction settings match the official AlphaFold2 notebook defaults checked on
2026-11-03. On Windows the runner automatically uses the installed WSL Python.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


_ROOT = Path(__file__).resolve().parents[1]
_ENV = _ROOT / "tools" / "localcolabfold" / ".venv"
_WEIGHTS = _ROOT / "tools" / "localcolabfold" / "weights"
_NOTEBOOK = "https://github.com/sokrypton/ColabFold/blob/84c27d9cc500489fd9b97545d2325b9d00f251d5/AlphaFold2.ipynb"
_DEFAULTS = {
    "num_models": 5, "num_recycles": 3,
    "recycle_early_stop_tolerance": None, "model_order": [1, 2, 3, 4, 5],
    "initial_guess": None, "num_ensemble": 1, "model_type": "auto",
    "msa_mode": "mmseqs2_uniref_env", "use_templates": False,
    "custom_template_path": None, "custom_template_cache_path": None,
    "num_relax": 0, "relax_max_iterations": 200, "relax_tolerance": 2.39,
    "relax_stiffness": 10.0, "relax_max_outer_iterations": 3,
    "keep_existing_results": False, "rank_by": "auto",
    "pair_mode": "unpaired_paired", "pairing_strategy": "greedy",
    "host_url": "https://api.colabfold.com", "random_seed": 0, "num_seeds": 1,
    "recompile_padding": 10, "zip_results": False,
    "save_single_representations": False, "save_pair_representations": False,
    "skip_output": [], "jobname_prefix": None, "save_all": False,
    "save_recycles": False, "use_dropout": False, "use_gpu_relax": False,
    "stop_at_score": 100.0, "dpi": 200, "max_seq": None, "max_extra_seq": None,
    "pdb_hit_file": None, "local_pdb_path": None, "use_cluster_profile": True,
    "calc_extra_ptm": False, "use_probs_extra": True,
    "max_template_date": "2100-01-01", "max_template_hits": 20,
    "use_fuse": True, "use_bfloat16": True, "use_fast_kernels": False,
    "kernel_backend": "auto", "compile_mode": "tuned", "max_msa": None,
}


def run_colabfold(input_path: str | Path, output_path: str | Path,
                 predict_structure: bool = True,
                 name: str | None = None) -> list[dict[str, str | None]]:
    """Generate webserver MSA and optionally predict with the local GPU.

    Args:
        input_path: FASTA (.fasta/.fa/.faa), sequence CSV/TSV, or an existing A3M.
            CSV/TSV must have id and sequence columns. Use ':' between chains.
            A FASTA with exactly two records is combined in memory into one
            query with ':' between the sequences, in their original order.
            No converted FASTA or CSV input file is written.
        output_path: Output DIRECTORY. Writes <query>_paired.a3m and
            <query>_unpaired.a3m, plus <query>.pdb for the best-ranked model
            when requested. Same-name MSA files are replaced.
            MSA-only leaves an existing PDB intact.
        predict_structure: False generates MSA only and does not require a GPU.
            True generates all five notebook-default models in a temporary
            directory, then exports only the top-ranked PDB.
        name: Optional output prefix. A single query produces <name>_paired.a3m,
            <name>_unpaired.a3m and optional <name>.pdb.
            Multiple queries use <name>_<query> as their prefix.
            Invalid filename characters become underscores. None keeps the
            FASTA filename stem (not its headers), or CSV/TSV/A3M query names.
            FASTA files with more than two records remain separate queries,
            named <filename>_1, <filename>_2, etc.
            Previous three-argument calls remain supported.

    Returns:
        One dictionary per query: name, paired_a3m and unpaired_a3m as absolute
        file paths, and pdb (None for MSA-only; no new structure was generated).
        Each category is saved as ONE file with ColabFold's official serializer.
        The complete unpaired category is kept together; no numbered per-chain
        files are created. Both files have a complete query and chain metadata.
        A category without homologs is represented by its query-only baseline.
        An A3M input is reused through ColabFold's own parser, without a web call.

    Model settings are fixed to notebook defaults: 5 models, 3 recycles, seed 0,
    one seed, auto MSA depth, no templates, no relaxation and no dropout.
    Local paths, network timeouts and output-file selection are local adaptations.
    The worker runs separately, so it does not change the caller's JAX/network state.
    """
    if not isinstance(predict_structure, bool):
        raise TypeError("predict_structure must be True or False")
    if name is not None and not isinstance(name, str):
        raise TypeError("name must be a string or None")
    prefix = _safe_name(name) if name is not None else None
    source = Path(input_path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError("input_path must be a file")
    output = Path(output_path).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise ValueError("output_path must be a directory")
    # Preflight input before starting the installed runtime or a web request.
    _name_queries(_read_queries(source), prefix)
    dispatch = _ROOT / ".cache" / "colabfold-dispatch"
    dispatch.mkdir(parents=True, exist_ok=True)
    worker_code = (
        "import importlib.util,json,sys; from pathlib import Path; "
        "s=importlib.util.spec_from_file_location('_colabfold_worker',sys.argv[1]); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        "r=m._run_local(Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4]=='1',json.loads(sys.argv[6])); "
        "Path(sys.argv[5]).write_text(json.dumps(r),encoding='utf-8')"
    )
    with tempfile.TemporaryDirectory(prefix="call-", dir=dispatch) as temporary:
        reply = Path(temporary) / "result.json"
        arguments = [str(Path(__file__).resolve()), str(source), str(output),
                     "1" if predict_structure else "0", str(reply)]
        if os.name == "nt":
            command = ["wsl.exe", "-d", os.environ.get("COLABFOLD_WSL_DISTRO", "Ubuntu"),
                       "--cd", str(_ROOT), "--exec", _wsl_path(str(_ENV / "bin" / "python")),
                       "-c", worker_code, *[_wsl_path(value) for value in arguments], json.dumps(prefix)]
        else:
            python = _ENV / "bin" / "python"
            if not python.exists():
                raise RuntimeError(f"Installed ColabFold environment is missing: {python}")
            command = [str(python), "-c", worker_code, *arguments, json.dumps(prefix)]
        subprocess.run(command, cwd=_ROOT, check=True)
        results = json.loads(reply.read_text(encoding="utf-8"))
    return [{"name": record["name"], "paired_a3m": str(output / record["paired_a3m"]),
             "unpaired_a3m": str(output / record["unpaired_a3m"]),
             "pdb": str(output / record["pdb"]) if record["pdb"] else None}
            for record in results]


def _wsl_path(value):
    match = re.match(r"^([A-Za-z]):[\\/](.*)$", value)
    if match:
        return "/mnt/" + match[1].lower() + "/" + match[2].replace("\\", "/")
    return value.replace("\\", "/")


def _safe_name(name):
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name.strip())
    if not name or name in (".", ".."):
        raise ValueError("Query names must be nonempty filenames")
    if len(name) > 160:
        name = name[:150] + "_" + hashlib.sha1(name.encode()).hexdigest()[:8]
    return name


def _name_queries(records, prefix):
    prefix = _safe_name(prefix) if prefix is not None else None
    renamed, seen = [], set()
    for query, sequence, msa in records:
        output_name = query if prefix is None else (
            prefix if len(records) == 1 else _safe_name(f"{prefix}_{query}")
        )
        filenames = [f"{output_name}_paired.a3m", f"{output_name}_unpaired.a3m",
                     f"{output_name}.pdb"]
        for filename in filenames:
            if filename.casefold() in seen:
                raise ValueError(f"Overlapping output filename: {filename}")
            seen.add(filename.casefold())
        renamed.append((output_name, sequence, msa))
    return renamed


def _read_queries(path):
    """Return (safe query name, sequence, optional existing A3M) records."""
    is_fasta = path.suffix.lower() in (".fasta", ".fa", ".faa")
    if path.suffix.lower() == ".a3m":
        _check_a3m(path)
        return [(_safe_name(path.stem), None, path)]
    if path.suffix.lower() in (".csv", ".tsv"):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t" if path.suffix.lower() == ".tsv" else ",")
            if not {"id", "sequence"}.issubset(reader.fieldnames or []):
                raise ValueError("CSV/TSV requires id and sequence columns")
            if {"a3mpath", "templatepath"}.intersection(reader.fieldnames or []):
                raise ValueError("Use sequence-only CSV/TSV, or an A3M file directly")
            queries = [(row["id"], row["sequence"], None) for row in reader]
    elif is_fasta:
        queries = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(">"):
                fields = line[1:].split()
                if not fields:
                    raise ValueError("FASTA contains an empty >header")
                queries.append((fields[0], "", None))
            elif queries:
                name, sequence, _ = queries[-1]
                queries[-1] = (name, sequence + line, None)
            else:
                raise ValueError("FASTA sequence appeared before its >header")
    else:
        raise ValueError("Input must be FASTA, CSV/TSV or A3M")
    normalized, seen = [], set()
    for index, (name, sequence, msa) in enumerate(queries, start=1):
        if is_fasta:
            name = path.stem if len(queries) <= 2 else f"{path.stem}_{index}"
        name = _safe_name(name or "")
        sequence = "".join((sequence or "").split()).upper()
        if not re.fullmatch(r"[A-Z]+(?::[A-Z]+)*", sequence):
            raise ValueError(f"Invalid amino-acid sequence: {name!r}")
        # Two FASTA chains intentionally share one output name; they are
        # combined below after each sequence has been validated separately.
        if name.casefold() in seen and not (is_fasta and len(queries) == 2):
            raise ValueError(f"Duplicate query filename: {name}")
        seen.add(name.casefold())
        normalized.append((name, sequence, msa))
    if not normalized:
        raise ValueError("Input contains no queries")
    if is_fasta and len(normalized) == 2:
        return [(normalized[0][0], ":".join(record[1] for record in normalized), None)]
    return normalized


def _check_a3m(path):
    sequence, started = [], False
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith(">"):
            if started:
                break
            started = True
        elif started and line and not line.startswith("#"):
            sequence.append(line)
    if not sequence or not re.fullmatch(r"[A-Za-z-]+", "".join(sequence)):
        raise ValueError(f"Invalid or empty A3M query: {path}")


def _configure_runtime(predict_structure):
    os.environ["JAX_PLATFORMS"] = "cuda,cpu" if predict_structure else "cpu"
    os.environ.setdefault("TF_FORCE_UNIFIED_MEMORY", "1")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("XDG_CACHE_HOME", str(_ROOT / ".cache"))
    import socket
    import requests
    import urllib3.util.connection
    if os.environ.get("COLABFOLD_FORCE_IPV4", "1") != "0":
        urllib3.util.connection.allowed_gai_family = lambda: socket.AF_INET
    original_request = requests.sessions.Session.request

    def request_with_timeout(self, method, url, **kwargs):
        kwargs["timeout"] = (30, 60)
        return original_request(self, method, url, **kwargs)

    requests.sessions.Session.request = request_with_timeout
    if predict_structure:
        import jax
        devices = jax.devices("gpu")
        if not devices or jax.default_backend() not in ("gpu", "cuda"):
            raise RuntimeError("CUDA GPU is unavailable; refusing CPU structure prediction")
        print(f"Local prediction GPU: {devices}", flush=True)


@contextmanager
def _server_lock():
    import fcntl
    lock = _ROOT / "tmp" / "msa-server.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _copy_atomic(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".colabfold-", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_msa_categories(result, work, name):
    """Save whole paired/unpaired categories using only ColabFold serialization."""
    from colabfold.input import msa_to_str
    unpaired, paired, sequences, cardinality, _ = result
    query_only = [f">{101 + index}\n{sequence}\n"
                  for index, sequence in enumerate(sequences)]
    has_rows = lambda msa: msa is not None and any(chunk.strip() for chunk in msa)
    categories = {
        "paired": (None, paired if has_rows(paired) else query_only),
        # The query-only baseline makes the whole complex query available in
        # the first row. All unpaired rows stay in this one category file.
        "unpaired": (unpaired if has_rows(unpaired) else None, query_only),
    }
    for category, (unpaired_rows, paired_rows) in categories.items():
        text = msa_to_str(unpaired_rows, paired_rows, sequences, cardinality)
        target = work / f"{name}_{category}.a3m"
        target.write_bytes(text.encode("utf-8"))
        _check_a3m(target)


@contextmanager
def _capture_msa_categories(work, name):
    """Capture upstream categories without classifying or slicing alignment rows."""
    import colabfold.batch as batch
    originals = {}

    def capture(provider):
        def wrapped(*args, **kwargs):
            result = provider(*args, **kwargs)
            _write_msa_categories(result, work, name)
            return result
        return wrapped

    try:
        for attribute in ("get_msa_and_templates", "unserialize_msa"):
            originals[attribute] = getattr(batch, attribute)
            setattr(batch, attribute, capture(originals[attribute]))
        yield
    finally:
        for attribute, provider in originals.items():
            setattr(batch, attribute, provider)


def _run_local(source, output, predict_structure, name=None):
    from importlib.metadata import version
    records = _name_queries(_read_queries(source), name)
    if version("colabfold") != "1.6.3":
        raise RuntimeError("Notebook defaults were verified for installed ColabFold 1.6.3")
    _configure_runtime(predict_structure)
    from colabfold.batch import get_queries, run, set_model_type
    from colabfold.download import download_alphafold_params
    from colabfold.utils import setup_logging
    work_root = _ROOT / ".cache" / "colabfold-runs"
    work_root.mkdir(parents=True, exist_ok=True)
    exported = []
    with tempfile.TemporaryDirectory(prefix="run-", dir=work_root) as temporary:
        for index, (name, sequence, existing_msa) in enumerate(records):
            work = Path(temporary) / str(index)
            work.mkdir()
            if existing_msa:
                query_path = work / f"{name}.a3m"
                shutil.copyfile(existing_msa, query_path)
                queries, is_complex = get_queries(query_path)
                queries = [(name, *query[1:]) for query in queries]
            else:
                # Match ColabFold's get_queries result without a CSV round trip.
                chains = sequence.split(":")
                is_complex = len(chains) > 1
                queries = [(name, chains if is_complex else chains[0], None, None)]
            options = dict(_DEFAULTS)
            options["model_order"] = list(_DEFAULTS["model_order"])
            options["skip_output"] = []
            options["model_type"] = set_model_type(is_complex, "auto")
            options["num_models"] = 5 if predict_structure else 0
            setup_logging(work / "log.txt")
            if predict_structure:
                download_alphafold_params(options["model_type"], _WEIGHTS)
            with _server_lock(), _capture_msa_categories(work, name):
                run(queries=queries, result_dir=work, is_complex=is_complex,
                    data_dir=_WEIGHTS, user_agent="colabfold/local-function/1.6.3",
                    prediction_callback=None, feature_dict_callback=None, **options)
            a3m = work / f"{name}.a3m"
            pdbs = list(work.glob(f"{name}_unrelaxed_rank_*.pdb"))
            best = list(work.glob(f"{name}_unrelaxed_rank_001_*.pdb"))
            if (not a3m.is_file() or a3m.stat().st_size == 0 or
                    (predict_structure and (len(pdbs) != 5 or len(best) != 1 or
                     not (work / f"{name}.done.txt").is_file() or
                     any(path.stat().st_size == 0 for path in pdbs)))):
                details = (work / "log.txt").read_text(encoding="utf-8")[-5000:]
                raise RuntimeError(f"ColabFold did not finish query {name}:\n{details}")
            _check_a3m(a3m)
            category_files = [work / f"{name}_{category}.a3m"
                              for category in ("paired", "unpaired")]
            for path in category_files:
                if not path.is_file():
                    raise RuntimeError(f"ColabFold MSA category was not captured: {path.name}")
                _check_a3m(path)
            for path in category_files:
                _copy_atomic(path, output / path.name)
            if predict_structure:
                _copy_atomic(best[0], output / f"{name}.pdb")
            exported.append({"name": name, "paired_a3m": f"{name}_paired.a3m",
                             "unpaired_a3m": f"{name}_unpaired.a3m",
                             "pdb": f"{name}.pdb" if predict_structure else None})
    return exported


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface; importing this module does not run it."""
    parser = argparse.ArgumentParser(
        description="Generate paired/unpaired A3M files and predict the best PDB with local ColabFold.",
        epilog=(
            "Exactly two FASTA records are joined in memory as one complex. "
            "FASTA output names default to the input filename stem, not its headers. "
            "On Windows, the installed WSL ColabFold environment is used automatically."
        ),
    )
    parser.add_argument(
        "input_path", type=Path,
        help="input FASTA (.fasta/.fa/.faa), sequence CSV/TSV, or existing A3M file",
    )
    parser.add_argument("output_path", type=Path, help="directory for exported A3M and PDB files")
    parser.add_argument(
        "--msa-only", action="store_true",
        help="generate only MSA files; by default also predict structure (requires a CUDA GPU)",
    )
    parser.add_argument(
        "--name", help="override the output prefix (default for FASTA: input filename without extension)",
    )
    args = parser.parse_args(argv)
    try:
        results = run_colabfold(
            args.input_path, args.output_path,
            predict_structure=not args.msa_only, name=args.name,
        )
    except subprocess.CalledProcessError as exc:
        parser.exit(1, f"{parser.prog}: error: ColabFold worker exited with status {exc.returncode}\n")
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"{parser.prog}: error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, f"{parser.prog}: interrupted\n")
    for record in results:
        print(f"{record['name']}:")
        for category in ("paired_a3m", "unpaired_a3m", "pdb"):
            if record[category] is not None:
                print(f"  {category}: {record[category]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
