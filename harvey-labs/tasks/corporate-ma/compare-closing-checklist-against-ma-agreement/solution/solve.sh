#!/bin/bash
# There is no reference solution for this task.
#
# Harvey LAB ships no golden files. The deliverable is a free-form document
# graded by an LLM against a 38-criterion rubric, and upstream's own published
# reference output for this task scores 36/38 -- which, under all-pass scoring,
# is 0.0. So there is nothing to copy in that would demonstrate a passing run.
#
# This script exists to make that explicit. Harbor's OracleAgent runs
# solution/solve.sh; rather than exit 0 and let an oracle run look like it
# succeeded while quietly scoring zero, fail loudly.

set -euo pipefail

cat >&2 <<'EOF'
No reference solution exists for this task.

Harvey LAB deliverables are free-form documents scored by an LLM judge against
a rubric; the benchmark ships no golden files, and no known output satisfies
all 38 criteria. Run this task with a real agent instead:

  harbor run -p <this task dir> -e docker \
    --agent agents.lab_harness:LABHarnessAgent \
    --model anthropic/claude-sonnet-4-6
EOF

exit 1
