# Grading

Maintainer notes for the graded suite. This file lives on the `tests` branch
only, alongside `tests/test_challenge.py` and `tests/conftest.py`, because it
says which check corresponds to which fix and where every threshold came from.
`main` carries `docs/automation.md`, which describes the workflows without any
of that.

## What is graded, and what is not

| Issue | Title | Checked by |
|---|---|---|
| #1 | The dark square is on the wrong end of the snake | 4 checks (2 snake lengths × 2 assertions) |
| #2 | One of the two hand-written players can no longer survive | 4 checks |
| #3 | Training teaches the snake to die | 3 checks |

The **write-up half of each "Done when" is not graded.** Issue #1 asks for the
regenerated board to be compared against the committed one by eye; issue #2 ends
with a question about *why* the trained network shrugs off a bug that destroys
the hand-written rule. A test can confirm the repo now supports those answers; it
cannot confirm they were honestly arrived at. Read the write-up — `pr-triage.yml`
flags a submission that never touched `README.md` for exactly this reason.

### Where each check comes from

Every one restates its issue's own "Done when" and nothing stricter, plus a guard
where the stated wording has a degenerate reading that would otherwise count as a
pass.

- **#1** one snake cell is overpainted at all (the guard: a plain gradient with no
  marker would leave the real check picking the largest of several equal steps,
  which is a coin flip) / that cell is `snake[0]`. Read off the rendered image by
  measuring how far the colour moves between neighbouring body cells: exactly one
  of those steps is large, and which end it sits at is the whole issue. Knows
  nothing about the marker's colour or the gradient's, so recolouring either
  still passes.
- **#2** three consecutive observation channels track free space at all (the
  guard) / they arrive in turn left, straight, turn right order / `food+safety`
  beats `greedy-food` / `food+safety` stops dying every episode. The order is
  measured, not assumed: an independent flood fill says how much room each move
  actually leads into, and the block of three channels correlating with it under
  any of the six orderings is the sensor.
- **#3** training leaves the snake dying *less* than an untrained policy on the
  same seed / it raises `eval_return` above that same baseline / a longer run
  grows the snake. Everything is relative to a freshly initialised policy, so
  there is no absolute number to argue with — which matters, because the issue
  describes a trained policy that is *worse than random*, and "worse than random"
  is a comparison.

### Four measurement notes

**The snake is staged long, not short.** The head marker is read off the colour
gradient, and the gradient is spread over however many cells the snake has — so
on a five-cell snake each step along the body is as large as the marker's own and
the picture genuinely cannot answer the question. At length 14 the marker is 6x
the median step on the shipped code and 17x once fixed. A check staged at length
6 fails its own guard.

**The hand-written players are handed the sensor block exactly as packed.**
Locating the block *and reordering it into action order* before handing it over
would quietly repair the bug before measuring it, and both rule checks would pass
on the unfixed repo. The location is measured so that a fork which moved the
block is still graded; the order is not touched, because the order is the thing
being graded.

**Issue #3's return bar is set from a measurement, not from the reference fix.**
Trained-minus-untrained over five seeds at 300 episodes: the shipped repo scores
−0.53 on return and *gains* 0.08 death rate; the one-character fix scores +8.51
and −0.57; the same fix with the return normalisation swapped for a plain mean
baseline scores +4.05 and −0.73. Deaths separate cleanly and returns do not — a
correct-but-slower fix can come out behind on return for an individual seed,
because it learns to stop crashing before it learns to eat and starving costs
the same −10 that crashing does. So the return check averages five seeds rather
than three and asks for +1.5, which sits between the two populations instead of
just above the broken one. At +2.0 on three seeds it rejects the mean-baseline
fix, which is a textbook variance reduction and unambiguously solves issue #3.

**Issue #1's reference GIF is rendered without the policy.** `docs/assets/head_marker.gif`
is a plain `py-eats-view --seed 74` run — the demo heuristic, which never
consults the policy or the free-space sensors. That is the point: its trajectory
is identical whether issues 2 and 3 are solved or not, so a candidate who has
fixed only the marker gets a frame-for-frame match. The trained-policy GIF cannot
do that job: `--seed 85` ends at length 28 with the sensors in order and at
length 12 without, so "identical to the committed GIF" would only hold once issue
#2 was solved too.

## Adding a check for a new issue

1. Write the test in `tests/test_challenge.py` and mark it
   `@pytest.mark.issue(N)`.
2. Add the issue title to `ISSUE_TITLES` in `tests/conftest.py`.
3. Confirm it **fails on `main` and passes on a real fix.** A check that does not
   fail on the shipped repo is not grading anything.
4. Then confirm it passes a fix that **looks nothing like the reference.** This
   is the step that is easy to skip and expensive to skip. Validating against
   fixes written by the same person who wrote the checks is how you end up
   grading your own solution rather than the issue.
   `reference_solutions/verify_variants.py` exists purely to attack that, and
   `reference_solutions/validate_checks.sh` runs it as one table. Both are
   gitignored and never published. It builds six trees and requires the last five
   to come out fully solved and the first to come out not solved at all:

   | variant | what it is |
   |---|---|
   | `unfixed` | the repository as shipped — the negative control |
   | `reference` | the one-line patch per issue |
   | `spelled-differently` | a different marker colour, explicit turn offsets, the minus moved onto the returns |
   | `corrected-downstream` | the buggy lines left alone and the results corrected after |
   | `rescaled-sensor` | the order fixed *and* the flood-fill cap changed |
   | `improved-while-fixing` | body drawn tail-first, sensors built by name, mean baseline instead of the return normalisation |
5. Assert only the issue's stated "Done when". Anything stricter fails honest work
   that solved the issue a different way.
