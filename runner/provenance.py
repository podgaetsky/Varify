"""Immutable reproducibility metadata tracker (standard library only).

Every run captures, into a read-only ``provenance.json`` **and** into the
run's telemetry payload:

* git commit hash, branch, dirty flag and remote (when inside a repo);
* the global random seed (auto-generated when not supplied, and applied to
  ``random`` — and ``numpy`` when importable — so the run is replayable);
* an environment state map: scheduler/threading variables verbatim, plus a
  SHA-256 digest of the *full* environment for tamper-evidence without
  leaking secrets;
* execution timestamps (UTC), host, user, interpreter, platform, argv, cwd;
* versions of the scientific stack packages that happen to be installed.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import random
import re
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from varify.utils.io_handlers import write_atomic

_ENV_WHITELIST = re.compile(
    r"^(SLURM_|PBS_|LSB_|OMP_|MKL_|OPENBLAS_|NUMEXPR_"
    r"|CUDA_VISIBLE_DEVICES$|VIRTUAL_ENV$|CONDA_DEFAULT_ENV$|HOSTNAME$)"
)
_TRACKED_PACKAGES = (
    "numpy", "scipy", "pandas", "matplotlib", "emcee", "corner", "yaml",
)


def _git_info() -> Dict[str, Any]:
    # Anchor to the repository holding the executing code, not the cwd.
    repo_dir = Path(__file__).resolve().parents[1]

    def _run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                ["git", *args], capture_output=True, text=True, timeout=10,
                cwd=repo_dir,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    commit = _run("rev-parse", "HEAD")
    if commit is None:
        return {"available": False}
    return {
        "available": True,
        "commit": commit,
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_run("status", "--porcelain")),
        "remote": _run("remote", "get-url", "origin"),
    }


def _package_versions() -> Dict[str, str]:
    from importlib import metadata

    versions: Dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        dist = "PyYAML" if name == "yaml" else name
        try:
            versions[name] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _environment_map() -> Dict[str, Any]:
    tracked = {
        k: v for k, v in sorted(os.environ.items()) if _ENV_WHITELIST.match(k)
    }
    digest = hashlib.sha256(
        json.dumps(sorted(os.environ.items())).encode("utf-8")
    ).hexdigest()
    return {
        "tracked": tracked,
        "full_env_sha256": digest,
        "variable_count": len(os.environ),
    }


def apply_seed(seed: Optional[int]) -> int:
    """Seed the global RNGs (stdlib ``random`` + numpy when present)."""
    if seed is None:
        seed = secrets.randbelow(2**31)
    random.seed(seed)
    try:
        import numpy as np  # type: ignore[import-untyped]

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    return seed


def capture_provenance(
    seed: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Capture the full reproducibility record (also applies the seed)."""
    applied_seed = apply_seed(seed)
    record: Dict[str, Any] = {
        "run_uuid": secrets.token_hex(8),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "timestamp_unix": time.time(),
        "seed": applied_seed,
        "git": _git_info(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "hostname": socket.gethostname(),
        },
        "user": _safe_user(),
        "argv": list(sys.argv),
        "cwd": str(Path.cwd()),
        "environment": _environment_map(),
        "packages": _package_versions(),
    }
    if extra:
        record["extra"] = extra
    return record


def _safe_user() -> str:
    try:
        return getpass.getuser()
    except (KeyError, OSError):
        return "unknown"


def write_provenance(record: Dict[str, Any], path: Path) -> Path:
    """Persist the record and mark the file read-only (immutability signal)."""
    p = write_atomic(path, json.dumps(record, indent=2, default=str))
    try:
        p.chmod(0o444)
    except OSError:
        pass
    return p
