"""SLURM integration: job submission, job registry and watchdog daemon."""

from varify.src.slurm.dispatcher import SlurmDispatcher
from varify.src.slurm.registry import JobRegistry

__all__ = ["SlurmDispatcher", "JobRegistry"]
