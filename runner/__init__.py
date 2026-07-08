"""Agnostic workflow runner with checkpointing, pre-flight and provenance."""

from varify.runner.checkpoint import CheckpointManager
from varify.runner.core import RunContext, RunSpec, WorkflowRunner, register_strategy
from varify.runner.preflight import PreflightError, PreflightReport, run_preflight
from varify.runner.provenance import capture_provenance

# Importing registers the built-in strategies.
from varify.runner import strategies as _strategies  # noqa: F401

__all__ = [
    "CheckpointManager",
    "PreflightError",
    "PreflightReport",
    "RunContext",
    "RunSpec",
    "WorkflowRunner",
    "capture_provenance",
    "register_strategy",
    "run_preflight",
]
