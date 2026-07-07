#!/bin/bash
# Per-case run script — executed by the scheduler inside the case directory.
# @JOB_NAME@ and parameter tokens are substituted by the framework.

set -euo pipefail

# Replace this stub with the real simulation invocation, e.g.:
#   srun ./my_simulation input
# The framework scrapes stdout.log for:
#   "RESULT: <float>"     (scan / optimizer metric)
#   "LOG_PROB: <float>"   (MCMC log-probability)
python3 - <<'EOF'
tau, gamma, kappa = @TAU@, @GAMMA@, @KAPPA@
result = (tau - 2.0) ** 2 + (gamma - 1.0) ** 2 + 0.1 * kappa
print(f"RESULT: {result:.8e}")
print(f"LOG_PROB: {-0.5 * result:.8e}")
EOF
