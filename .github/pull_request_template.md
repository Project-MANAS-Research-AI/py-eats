## Which issues does this address?

<!-- e.g. Closes #2, #3. List only what you actually attempted. Solving one of
     the three is a perfectly good pull request. -->

## What did you change, and why?

<!-- One short paragraph per issue. The reasoning matters more than the diff;
     the diff is right there. -->

## Numbers

<!-- Parts of each "Done when" are read rather than tested, so they only count
     if they are here.

     #1 — say that your regenerated board matches `docs/assets/head_marker.gif`,
          and how you checked
     #2 — the `py-eats-bench --rules` table: mean length and died% for both
          hand-written players, before and after
     #3 — the last few `py-eats-train` log lines, showing eval_return and
          deaths

     Say which seed and how many episodes. "It improved" is not a number. -->

## Issue 2's closing question

<!-- "This bug costs the trained network almost nothing but destroys the
     hand-written rule. Why can the network shrug it off when the rule
     cannot?"

     A few sentences. This is read by a human and is not graded by any test. -->

---

<!-- Please leave the rest of this in place.

     You are graded against this repo's copy of the tests, not the copy in your
     branch — yours is replaced before anything runs. Editing tests/,
     .github/, pyproject.toml, uv.lock, .python-version or docs/assets/ cannot
     change your result and will get this pull request closed automatically,
     before anyone reads it. Neither can editing src/snake_rl/benchmark.py,
     which issue 2 asks you to leave alone. Nor does anything here need a file
     deleted — a pull request that removes files is closed too.

     Two other things get a branch closed automatically: code that does not
     parse, and code that fails the guard-rail suite. Both are worth five
     seconds of your own time before you push:

         uv run ruff check --select E9,F63,F7,F82 .
         uv run pytest -m "not challenge" -q

     Everything else reaches a reviewer, including a submission that solved
     nothing yet. The run posts a per-issue scorecard — read that rather than
     the red or green tick. -->

- [ ] `uv run pytest` passes on my fork (the guard rails, not the graded checks)
- [ ] I have not modified `tests/`, `.github/`, the environment files, `docs/assets/` or `src/snake_rl/benchmark.py`
- [ ] I have not deleted any files
- [ ] The numbers above say which seed and how many episodes
