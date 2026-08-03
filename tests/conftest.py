"""Turns the graded test results into a per-issue scorecard.

Locally it prints a short table at the end of the run. In GitHub Actions it
also writes the same table to the job summary, so a run can be read without
opening the log, and -- if ``CHALLENGE_SCORECARD_JSON`` names a path -- the
same result as JSON, which is how ``pr-checks.yml`` compares a pull request
against the branch it targets.

The point of the table is that a failing run is still a *readable* run. Three
issues are open at once, a submission normally fixes them one at a time, and
"14 failed" tells nobody which of the three moved.
"""

import json
import os

import matplotlib

# Before anything imports viewer.py, which builds a figure at construction
# time and would otherwise go looking for a display that CI does not have.
matplotlib.use("Agg")

ISSUE_TITLES = {
    1: "The dark square is on the wrong end of the snake",
    2: "One of the two hand-written players can no longer survive",
    3: "Training teaches the snake to die",
}

_RESULTS = {}
_ISSUE_BY_NODE = {}
_COLLECTION_ERRORS = []


def pytest_configure(config):
  config.addinivalue_line(
      "markers", "issue(number): the challenge issue this test grades")


def pytest_collection_modifyitems(items):
  for item in items:
    for marker in item.iter_markers("issue"):
      _ISSUE_BY_NODE[item.nodeid] = marker.args[0]


def pytest_collectreport(report):
  """Catch a suite that never even imported.

  If `src/snake_rl/env.py` does not parse, or `SnakeEnv` was renamed, pytest stops
  at collection and no test runs at all -- so without this the run ends with
  no scorecard whatsoever, which is the least useful thing it could do when
  someone is trying to find out where a submission stands.
  """
  if report.failed:
    _COLLECTION_ERRORS.append(report.nodeid or "<the test session>")


def pytest_runtest_logreport(report):
  number = _ISSUE_BY_NODE.get(report.nodeid)
  if number is None:
    return
  # A test whose fixture blew up never reaches the call phase, so count that
  # as a failure rather than losing the issue from the scorecard entirely.
  if report.when == "call" or (report.when == "setup" and not report.passed):
    _RESULTS.setdefault(number, []).append(report.passed)


def _scorecard():
  """Every issue, every time.

  An issue with no result is reported as such rather than dropped. "This was
  not graded" and "this did not pass" are different things, and a maintainer
  reading the table needs to be able to tell them apart.
  """
  lines = ["| Issue | Title | Result |", "|---|---|---|"]
  for number in sorted(ISSUE_TITLES):
    outcomes = _RESULTS.get(number)
    if not outcomes:
      result = "could not be graded"
    else:
      result = "solved" if all(outcomes) else "not solved"
      result += f" ({sum(outcomes)}/{len(outcomes)} checks)"
    lines.append(f"| #{number} | {ISSUE_TITLES[number]} | {result} |")

  ungraded = sorted(set(ISSUE_TITLES) - set(_RESULTS))
  if ungraded:
    lines.append("")
    if _COLLECTION_ERRORS:
      lines.append(
          "**Nothing could be collected from "
          + ", ".join(f"`{where}`" for where in sorted(set(_COLLECTION_ERRORS)))
          + ".** Usually that means one of `src/snake_rl/env.py`, "
            "`src/snake_rl/train.py` or `src/snake_rl/viewer.py` does not "
            "import at all -- a syntax error, or a name that moved. Fix that "
            "and the issues above can actually be graded.")
    else:
      lines.append("Not graded in this run: "
                   + ", ".join(f"#{number}" for number in ungraded))
  return "\n".join(lines)


def _solved_by_issue():
  """Counts rather than a bare pass/fail, so partial progress is visible.

  `compare_scorecards.py` needs to tell "solved two of the three checks on
  issue #2, up from none" apart from "still not solved" -- a submission that
  moved an issue forward without finishing it has done something, and a bool
  throws that away.
  """
  return {str(number): {"passed": sum(outcomes), "total": len(outcomes)}
          for number, outcomes in _RESULTS.items()}


def pytest_terminal_summary(terminalreporter):
  json_path = os.environ.get("CHALLENGE_SCORECARD_JSON")
  if json_path:
    # Written even for an empty run, so a caller comparing two runs can tell
    # "graded nothing" apart from "the file never appeared".
    with open(json_path, "w", encoding="utf-8") as handle:
      json.dump(_solved_by_issue(), handle, indent=2, sort_keys=True)

  # Nothing collected and nothing broken means the graded tests were skipped
  # on purpose -- `pytest -m "not challenge"`, which is how the guard rails
  # are run on their own. Printing "could not be graded" there is a lie about
  # a run that was never asked to grade anything, and in a CI job summary it
  # lands above the real scorecard and reads like a failure.
  if not _RESULTS and not _COLLECTION_ERRORS:
    return

  # Otherwise print even when nothing ran: a run that collapsed on import is
  # exactly when you most want to be told what happened, in the same place
  # you would have read the result.
  table = _scorecard()
  terminalreporter.write_sep("=", "challenge scorecard")
  terminalreporter.write_line(table)
  summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
  if summary_path:
    with open(summary_path, "a", encoding="utf-8") as handle:
      handle.write("## Challenge scorecard\n\n" + table + "\n")
