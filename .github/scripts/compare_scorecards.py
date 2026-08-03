"""Compare two challenge scorecards: reject a regression, and -- when the
pull request touches graded code -- require it to move something forward.

Usage: ``compare_scorecards.py BASE_JSON HEAD_JSON``

Both files are written by ``tests/conftest.py`` when
``CHALLENGE_SCORECARD_JSON`` is set::

    {"1": {"passed": 4, "total": 4}, "2": {"passed": 1, "total": 4}, ...}

Counts rather than a bare pass/fail, so a submission that got two of an
issue's four checks passing has visibly done something even though the issue
is not finished.

Two rules, and they answer different questions:

* **Nothing may regress.** An issue passing on the base branch that stops
  passing here fails the run. Solving nothing is fine -- plenty of honest pull
  requests (docs, CI, dependency bumps) solve nothing. Breaking something is
  not.
* **A pull request that rewrites graded code has to move an issue.** Without
  that, editing the environment at random is indistinguishable from doing the
  work: nothing regressed, so nothing failed. Set by ``MUST_FIX=true`` from
  the workflow, which decides it from the diff rather than from the labels.

Restored from the `tests` branch by pr-checks.yml alongside the suite itself, so
a pull request cannot rewrite the thing that judges it.
"""

import json
import os
import sys


def read(path):
  try:
    with open(path, encoding="utf-8") as handle:
      return json.load(handle)
  except (OSError, ValueError):
    return {}


def counts(entry):
  """``(passed, total)`` from either scorecard shape.

  Older runs wrote a bare boolean per issue. Reading both means a comparison
  against a base branch graded before this change still works instead of
  reporting every issue as ungraded.
  """
  if entry is None:
    return None
  if isinstance(entry, bool):
    return (1, 1) if entry else (0, 1)
  return int(entry.get("passed", 0)), int(entry.get("total", 0))


def label(entry):
  pair = counts(entry)
  if pair is None:
    return "not graded"
  passed, total = pair
  if total and passed == total:
    return f"solved ({passed}/{total})"
  return f"not solved ({passed}/{total})"


def solved(pair):
  return pair is not None and pair[1] > 0 and pair[0] == pair[1]


def main(argv):
  if len(argv) != 3:
    sys.exit(f"usage: {argv[0]} BASE_JSON HEAD_JSON")
  base, head = read(argv[1]), read(argv[2])
  base_ref = os.environ.get("BASE_REF", "the base branch")
  must_fix = os.environ.get("MUST_FIX", "").lower() == "true"

  if not base and not head:
    sys.exit("::error::Neither run produced a scorecard, so nothing was "
             "graded. Treating that as a failure rather than a pass.")

  rows = [f"| Issue | {base_ref} | This PR |", "|---|---|---|"]
  regressed, progressed = [], []
  for issue in sorted(set(base) | set(head), key=int):
    was, now = counts(base.get(issue)), counts(head.get(issue))
    rows.append(f"| #{issue} | {label(base.get(issue))} | {label(head.get(issue))} |")

    if was is None or now is None:
      continue
    # Either half of an issue going backwards counts: fully un-solving it, or
    # passing fewer of its checks than before.
    if (solved(was) and not solved(now)) or now[0] < was[0]:
      regressed.append(issue)
    if (solved(now) and not solved(was)) or now[0] > was[0]:
      progressed.append(issue)

  table = "\n".join(rows)
  print(table)
  summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
  if summary_path:
    with open(summary_path, "a", encoding="utf-8") as handle:
      handle.write("## Issues, before and after this PR\n\n" + table + "\n")

  if regressed:
    sys.exit("::error::This PR un-solves "
             + ", ".join("issue #" + number for number in regressed)
             + f", which passed on {base_ref} before it.")

  moved = ", ".join("#" + number for number in progressed) or "none"
  print(f"moved forward: {moved}")

  if must_fix and not progressed:
    sys.exit(
        "::error::This PR rewrites code the challenge is graded on but moves "
        "no issue forward -- every check that failed on " + base_ref + " "
        "still fails. If it is meant to fix something, the scorecard above is "
        "where that would show. If it is not (a refactor, a comment, a "
        "rename), say so and a maintainer can label it `ci-change`.")
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv))
