# /// script
# dependencies = []
# ///
#
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset-level metric for the Harvey LAB port.

Harvey LAB reports Pass@1: the mean of per-task all-pass scores, where each
task scores 1.0 only if every rubric criterion passed. Since the verifier has
already collapsed each task to 0.0 or 1.0, Pass@1 is just the mean.

Reads the rewards JSONL Harbor produces (one object per trial) and writes a
JSON object of aggregate metrics.
"""

import argparse
import json
from pathlib import Path


def main(input_path: Path, output_path: Path) -> None:
    scores: list[float] = []
    n_criteria_total = 0
    n_criteria_passed = 0

    for line in input_path.read_text().splitlines():
        if not line.strip():
            continue
        reward = json.loads(line)

        # A null entry means the trial never produced a reward (infra
        # failure). Upstream scores an absent deliverable as a task failure,
        # so count it as 0 rather than dropping it from the denominator.
        if reward is None:
            scores.append(0.0)
            continue

        # The verifier writes several keys; `reward` is the task score. Fall
        # back to the sole value for single-key rewards from other producers.
        if "reward" in reward:
            scores.append(float(reward["reward"]))
        elif len(reward) == 1:
            scores.append(float(next(iter(reward.values()))))
        else:
            raise ValueError(
                f"Reward object has no 'reward' key and is not single-valued: "
                f"{sorted(reward)}"
            )

        n_criteria_total += int(reward.get("n_criteria", 0) or 0)
        n_criteria_passed += int(reward.get("n_passed", 0) or 0)

    n_tasks = len(scores)
    pass_at_1 = sum(scores) / n_tasks if n_tasks else 0.0

    metrics = {
        "pass_at_1": pass_at_1,
        "n_tasks": n_tasks,
        "n_tasks_passed": sum(1 for s in scores if s >= 1.0),
    }

    # Criterion-level pass rate is not the headline number, but it separates
    # "missed one criterion" from "missed thirty" across a failing set.
    if n_criteria_total:
        metrics["criterion_pass_rate"] = n_criteria_passed / n_criteria_total
        metrics["n_criteria"] = n_criteria_total
        metrics["n_criteria_passed"] = n_criteria_passed

    output_path.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-path",
        type=Path,
        required=True,
        help="Path to a jsonl file containing rewards, one json object per line.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        required=True,
        help="Path to a json file where the metric will be written as a json object.",
    )
    args = parser.parse_args()
    main(args.input_path, args.output_path)
