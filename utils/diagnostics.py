"""Cluster post-mortem engine (standard library only).

Parses SLURM stdout/stderr logs for failure signatures and coarse
performance metrics:

* **Error flagging** — a regex catalog classifies Out-Of-Memory kills,
  wall-time cancellations, node failures, segfaults, Python tracebacks,
  NaN contamination, MPI aborts, disk-full and permission errors.
* **Performance profiling** — first/last embedded timestamps give an
  estimated wall time; progress-line counting gives an iteration
  throughput; ``time(1)`` real/user/sys lines are extracted when present.

Use ``analyze_log`` for one file, ``post_mortem`` for one job directory, or
``profile_workspace`` to sweep every case folder into a CSV summary.
"""

from __future__ import annotations

import csv
import datetime as _dt
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from utils.io_handlers import read_text_safe

# ── Failure signature catalog ─────────────────────────────────────────────────

FAILURE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("oom", re.compile(
        r"oom[-_ ]?kill|out[ -]of[ -]memory|MemoryError"
        r"|Exceeded job memory limit|Cannot allocate memory", re.I)),
    ("timeout", re.compile(
        r"DUE TO TIME LIMIT|CANCELLED AT .* DUE TO TIME"
        r"|walltime .* exceeded", re.I)),
    ("node_failure", re.compile(r"NODE_FAIL|node failure|Transient node", re.I)),
    ("segfault", re.compile(
        r"Segmentation fault|signal 11|SIGSEGV|core dumped", re.I)),
    ("python_error", re.compile(r"Traceback \(most recent call last\)")),
    ("nan", re.compile(r"(?<![A-Za-z])nan(?![A-Za-z])", re.I)),
    ("mpi_abort", re.compile(r"MPI_ABORT|MPI_Abort|PMIx.*abort", re.I)),
    ("disk_full", re.compile(r"No space left on device|Disk quota exceeded", re.I)),
    ("permission", re.compile(r"Permission denied|Operation not permitted", re.I)),
    ("killed", re.compile(r"^Killed$|Killed process", re.I | re.M)),
]

_SEVERITY: Dict[str, str] = {
    "oom": "fatal", "timeout": "fatal", "node_failure": "fatal",
    "segfault": "fatal", "mpi_abort": "fatal", "disk_full": "fatal",
    "killed": "fatal", "python_error": "error", "permission": "error",
    "nan": "warning",
}

# ── Timestamp / throughput extraction ─────────────────────────────────────────

_TS_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})"),
     "%Y-%m-%d %H:%M:%S"),
    (re.compile(r"^\[?(\d{2}:\d{2}:\d{2})\]?"), "%H:%M:%S"),
]
_PROGRESS_RE = re.compile(r"\b(step|iter(?:ation)?|epoch|cycle)\b", re.I)
_TIME1_RE = re.compile(
    r"^(?:real|user|sys)\s+(\d+)m([\d.]+)s\s*$", re.M
)


@dataclass
class LogFinding:
    """One classified failure signature hit inside a log file."""

    category: str
    severity: str
    file: str
    line_no: int
    excerpt: str


@dataclass
class PostMortemReport:
    """Aggregated diagnosis of one job directory."""

    job_dir: str
    findings: List[LogFinding] = field(default_factory=list)
    wall_time_s: Optional[float] = None
    progress_lines: int = 0
    progress_rate_hz: Optional[float] = None
    time1_real_s: Optional[float] = None
    log_bytes: int = 0
    log_lines: int = 0
    verdict: str = "clean"

    @property
    def failed(self) -> bool:
        return self.verdict not in ("clean", "warning")

    def render(self) -> str:
        lines = [
            f"Post-mortem: {self.job_dir}",
            f"  verdict        : {self.verdict}",
            f"  wall time (est): "
            f"{'-' if self.wall_time_s is None else f'{self.wall_time_s:.1f}s'}",
            f"  progress lines : {self.progress_lines}"
            + (f"  ({self.progress_rate_hz:.3g}/s)"
               if self.progress_rate_hz else ""),
            f"  log size       : {self.log_lines} lines / {self.log_bytes} B",
        ]
        for f_ in self.findings:
            lines.append(
                f"  [{f_.severity.upper():7s}] {f_.category:12s} "
                f"{f_.file}:{f_.line_no}  {f_.excerpt[:100]}"
            )
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
#  Single-file analysis
# ═════════════════════════════════════════════════════════════════════════════

def _parse_timestamps(text: str) -> Optional[float]:
    """Estimate elapsed seconds between first and last embedded timestamp."""
    for pattern, fmt in _TS_PATTERNS:
        hits = pattern.findall(text)
        if len(hits) < 2:
            continue
        try:
            if isinstance(hits[0], tuple):
                first = _dt.datetime.strptime(" ".join(hits[0]), fmt)
                last = _dt.datetime.strptime(" ".join(hits[-1]), fmt)
            else:
                first = _dt.datetime.strptime(hits[0], fmt)
                last = _dt.datetime.strptime(hits[-1], fmt)
        except ValueError:
            continue
        delta = (last - first).total_seconds()
        if delta < 0 and fmt == "%H:%M:%S":  # midnight rollover
            delta += 86400.0
        if delta >= 0:
            return delta
    return None


def analyze_log(
    path: Union[str, Path],
    max_findings_per_category: int = 3,
) -> List[LogFinding]:
    """Classify every failure signature found in one log file."""
    p = Path(path)
    findings: List[LogFinding] = []
    counts: Dict[str, int] = {}
    try:
        text = read_text_safe(p)
    except FileNotFoundError:
        return findings
    for line_no, line in enumerate(text.splitlines(), 1):
        for category, pattern in FAILURE_PATTERNS:
            if counts.get(category, 0) >= max_findings_per_category:
                continue
            if pattern.search(line):
                counts[category] = counts.get(category, 0) + 1
                findings.append(LogFinding(
                    category=category,
                    severity=_SEVERITY[category],
                    file=p.name,
                    line_no=line_no,
                    excerpt=line.strip(),
                ))
    return findings


# ═════════════════════════════════════════════════════════════════════════════
#  Job-directory post-mortem & workspace sweep
# ═════════════════════════════════════════════════════════════════════════════

def post_mortem(
    job_dir: Union[str, Path],
    log_names: Sequence[str] = ("stdout.log", "stderr.log"),
) -> PostMortemReport:
    """Diagnose one job directory: failure flags + coarse performance profile."""
    d = Path(job_dir)
    report = PostMortemReport(job_dir=str(d))
    all_text: List[str] = []
    for name in log_names:
        log_path = d / name
        if not log_path.exists():
            continue
        text = read_text_safe(log_path)
        all_text.append(text)
        report.log_bytes += log_path.stat().st_size
        report.log_lines += text.count("\n") + (1 if text else 0)
        report.findings.extend(analyze_log(log_path))

    joined = "\n".join(all_text)
    report.wall_time_s = _parse_timestamps(joined)
    report.progress_lines = sum(
        1 for line in joined.splitlines() if _PROGRESS_RE.search(line)
    )
    if report.wall_time_s and report.progress_lines:
        report.progress_rate_hz = report.progress_lines / report.wall_time_s
    time1 = _TIME1_RE.findall(joined)
    if time1:
        m, s = time1[0]
        report.time1_real_s = int(m) * 60 + float(s)

    severities = {f.severity for f in report.findings}
    if "fatal" in severities:
        report.verdict = "fatal"
    elif "error" in severities:
        report.verdict = "error"
    elif "warning" in severities:
        report.verdict = "warning"
    elif not all_text:
        report.verdict = "no_logs"
    return report


def profile_workspace(
    root: Union[str, Path],
    csv_out: Optional[Union[str, Path]] = None,
    log_names: Sequence[str] = ("stdout.log", "stderr.log"),
) -> List[PostMortemReport]:
    """Post-mortem every subdirectory of *root*; optionally dump a CSV table."""
    root_p = Path(root)
    reports = [
        post_mortem(d, log_names)
        for d in sorted(root_p.iterdir())
        if d.is_dir()
    ]
    if csv_out is not None:
        out = Path(csv_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "job_dir", "verdict", "wall_time_s", "progress_lines",
                "progress_rate_hz", "log_lines", "log_bytes", "categories",
            ])
            for r in reports:
                writer.writerow([
                    r.job_dir, r.verdict,
                    "" if r.wall_time_s is None else f"{r.wall_time_s:.1f}",
                    r.progress_lines,
                    "" if r.progress_rate_hz is None
                    else f"{r.progress_rate_hz:.4g}",
                    r.log_lines, r.log_bytes,
                    ";".join(sorted({f.category for f in r.findings})),
                ])
    return reports
