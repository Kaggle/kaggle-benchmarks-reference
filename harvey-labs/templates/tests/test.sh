#!/bin/bash
# Harbor verifier for the Harvey LAB Harbor port.
#
# Grades /workspace/output against task.json's rubric with an LLM judge, then
# writes Harbor's reward file. Scoring is all-pass: 1.0 only if every one of
# the criteria passes, else 0.0 -- no partial credit, matching upstream.
#
# Deliberately not `set -e`. A missing reward file is an infrastructure error
# in Harbor, not a task failure, so every exit path must leave one behind; the
# trap below guarantees that even if the judge crashes.

set -uo pipefail

VERIFIER_DIR=/logs/verifier
REWARD_JSON="$VERIFIER_DIR/reward.json"
SCORES_JSON="$VERIFIER_DIR/scores.json"
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR=/workspace/output

mkdir -p "$VERIFIER_DIR"

# Always leave a reward behind. Overwritten on success by judge.py; keep the
# keys in lockstep with that file's build_reward().
write_zero_reward() {
  if [ ! -s "$REWARD_JSON" ]; then
    cat > "$REWARD_JSON" <<'EOF'
{
  "reward": 0.0,
  "score": 0.0,
  "n_criteria": 0,
  "n_passed": 0,
  "judge_latency_ms": 0.0,
  "n_judge_errors": 0
}
EOF
  fi
}
trap write_zero_reward EXIT

if [ ! -d "$OUTPUT_DIR" ]; then
  echo "No output directory at $OUTPUT_DIR — agent produced nothing. Score 0." >&2
  exit 0
fi

echo "Output files:"
ls -la "$OUTPUT_DIR" || true

# Every dependency in judge.py's inline block is baked into the image, so the
# judge runs against the system interpreter. Nothing is installed at verify
# time, which is what keeps the verifier's allowlist down to the model
# endpoints -- no package index has to be reachable from in here.
python3 "$TESTS_DIR/judge.py" \
  --task-json "$TESTS_DIR/task.json" \
  --prompt "$TESTS_DIR/rubric_criterion.txt" \
  --output-dir "$OUTPUT_DIR" \
  --reward-json "$REWARD_JSON" \
  --scores-json "$SCORES_JSON"
judge_status=$?

if [ "$judge_status" -eq 3 ]; then
  # A judge that never answered. Its criteria would all read as failures, so
  # judge.py refuses to write a score; the trap's zero reward stands, which
  # reads as an infrastructure problem rather than a graded task.
  echo "A configured judge was unreachable; no score written. See above." >&2
elif [ "$judge_status" -ne 0 ]; then
  echo "Judge failed with exit code $judge_status; scoring 0." >&2
fi

# Exit 0 regardless: the reward file carries the verdict. A nonzero exit here
# would read as a broken verifier rather than a failed task.
exit 0
