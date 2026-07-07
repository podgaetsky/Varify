#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  Varify — generalized SLURM dispatch template
#
#  Cluster portability layer: every site-specific value is either an
#  @TOKEN@ placeholder (resolved from the RunSpec `slurm:` mapping at
#  submission time — #SBATCH lines cannot read environment variables) or a
#  VARIFY_* environment variable with a safe default (resolved at job
#  start).  Retargeting partitions, resources or environment paths touches
#  ONLY this file or that mapping — never the Python execution logic.
#
#  Tokens: @JOB_NAME@ @PARTITION@ @TIME@ @NODES@ @NTASKS@ @CPUS_PER_TASK@
#          @MEM@ @EXTRA_DIRECTIVES@ @PYTHON_BIN@ @SCRIPT@ @SCRIPT_ARGS@
#          @WORKDIR@ @RUN_DIR@
# ═══════════════════════════════════════════════════════════════════════════
#SBATCH --job-name=@JOB_NAME@
#SBATCH --partition=@PARTITION@
#SBATCH --time=@TIME@
#SBATCH --nodes=@NODES@
#SBATCH --ntasks=@NTASKS@
#SBATCH --cpus-per-task=@CPUS_PER_TASK@
#SBATCH --mem=@MEM@
#SBATCH --output=@RUN_DIR@/slurm-%j.out
#SBATCH --error=@RUN_DIR@/slurm-%j.err
#SBATCH --signal=B:USR1@60
@EXTRA_DIRECTIVES@

set -euo pipefail

# ── Site environment (override via environment, not by editing code) ────────
export VARIFY_WORKDIR="${VARIFY_WORKDIR:-@WORKDIR@}"
export VARIFY_PYTHON="${VARIFY_PYTHON:-@PYTHON_BIN@}"
export VARIFY_ENV_SETUP="${VARIFY_ENV_SETUP:-}"   # e.g. "module load python/3.11"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

# Re-entry guard: tells WorkflowRunner it is already inside the allocation,
# so the workflow executes locally instead of resubmitting itself.
export VARIFY_INSIDE_SLURM=1

if [[ -n "${VARIFY_ENV_SETUP}" ]]; then
    eval "${VARIFY_ENV_SETUP}"
fi

cd "${VARIFY_WORKDIR}"
echo "[varify] host=$(hostname) job=${SLURM_JOB_ID:-n/a} started $(date -u +%FT%TZ)"

# `exec` keeps Python as the signalled process so the checkpoint layer
# receives SLURM's wall-time warning (USR1) and pre-kill TERM directly.
exec "${VARIFY_PYTHON}" "@SCRIPT@" @SCRIPT_ARGS@
