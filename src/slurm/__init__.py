"""SLURM integration: job submission, job registry and watchdog daemon."""

from src.slurm.dispatcher import SlurmDispatcher
from src.slurm.registry import JobRegistry

__all__ = ["SlurmDispatcher", "JobRegistry"]
