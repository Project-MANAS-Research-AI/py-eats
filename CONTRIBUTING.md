# How to submit

Everything you need to do is on the **Issues** tab. This page is only about
the mechanics of getting your work from your machine into a pull request here.

All three issues are code, and they are submitted as **one pull request** from
your fork. One pull request can address any number of them. You do not need a
separate one per issue, and you do not need to attempt all three.

---

## 1. Fork the repository

Press **Fork** at the top right of this repo's page and create the fork under
your own account. Leave the name as it is.

You now have your own copy at `https://github.com/<your-username>/py-eats`.
All of your work happens there. You cannot push to this repository directly,
and you do not need to.

## 2. Clone your fork and set it up

```bash
git clone https://github.com/<your-username>/py-eats.git
cd py-eats
```

Add this repository as a second remote called `upstream`, so you can pull in
any changes made after you forked:

```bash
git remote add upstream https://github.com/Project-MANAS-Research-AI/py-eats.git
git remote -v      # origin = your fork, upstream = here
```

Then set up the environment. One command does all of it — creates `.venv`,
installs the versions pinned in `uv.lock`, and installs the project so
`import snake_rl` works from anywhere:

```bash
uv sync
```

(No uv? `pip install uv`, or see https://docs.astral.sh/uv/. Everything below
is `uv run <command>`, which uses that environment without you activating it.)

Check it runs before you change anything:

```bash
uv run py-eats-train --episodes 20
```

## 3. Read the issues properly

Read the whole issue, including the **Done when** line at the bottom. That
line is the definition of finished — not "I changed the environment", but the
specific, checkable thing the issue asks you to end up with.

**The whole shipped test suite comes out green with all three bugs present.** `pytest`
will not find them for you. The committed `docs/assets/head_marker.gif`,
`docs/assets/trained_policy.gif`, `docs/assets/training_curve.png` and the
README's table are the ground truth.

Each issue asks you to *measure* something and *report the number*. Those
numbers are part of the answer. An issue is not finished without them.

## 4. Make a branch

Do not work on `main` directly, even on your own fork. Branch off it:

```bash
git checkout main
git pull upstream main
git checkout -b fix/head-marker-and-sensors
```

Name it after what you are doing. `fix/head-marker-and-sensors`,
`issue-3-reinforce-sign`, and `sensor-order` are all fine. `patch-1` is not.

## 5. Do the work

**What you are expected to change:**

| Path | What it is |
|---|---|
| `src/snake_rl/env.py` | the game: movement, reward, observations |
| `src/snake_rl/models.py` | the policy network |
| `src/snake_rl/train.py` | the training loop |
| `src/snake_rl/viewer.py` | the visualisation |
| `README.md` | your write-up: what you changed, why, and your numbers |

**What you must not change:**

| Path | Why |
|---|---|
| `src/snake_rl/benchmark.py` | issue 2 says so. The two players are correct; what they are handed is not. |
| `tests/` | these are the checks. Changing them does not change your result. |
| `.github/` | the automation |
| `pyproject.toml`, `uv.lock`, `.python-version` | these decide what your code is even run on, so everyone is checked on the same one |
| `.gitignore` | |
| `docs/assets/` | the ground truth you are comparing against |

A pull request that touches any of those is **closed automatically**, before
anyone reads it. So is one that **deletes** files — nothing here is solved by
removing anything, so a deletion is either an accident or an attempt to make a
failing check go away. If you think one of these files is genuinely wrong, say
so in a comment on the relevant issue rather than editing it.

Closed is not the same as rejected: you get a comment saying exactly what
tripped, and pressing **Reopen** after fixing the branch runs everything again.

`src/snake_rl/benchmark.py` is the one worth being explicit about. Issue 2's
symptom is a hand-written player that dies every episode, and you can make that
symptom go away by rewriting the player until it copes with the numbers it is
handed. That is not the fix — the numbers are wrong, and every other reader of
them, including the trained network, is still being misled.

## 6. Check your work locally

```bash
uv run pytest
```

This is a set of sanity checks: that the game still returns finite
observations of the size it advertises, still terminates, still responds to
the action, and above all **still produces the same trajectory from the same
seed**. It says nothing about whether you solved anything.

Keep it green. It is easy to "fix" an issue by accident — by making the game
stop being reproducible, or letting a number run to infinity — and then every
measurement you report means nothing.

## 7. Commit

Commit in steps that make sense on their own, with messages that say what
changed and why:

A subject line saying what changed, a blank line, then the part worth reading:

```
Pack the free-space sensors in action order

Name what each reading was lining up with before and what it lines up
with now, and say what that did to anything steering by them. Then the
number you measured, so the claim can be checked.
```

`update`, `fix`, `final`, and `asdf` tell a reviewer nothing. The diff already
says *what* changed; the message is where you say *why*.

## 8. Push to your fork

```bash
git push -u origin fix/head-marker-and-sensors
```

## 9. Open the pull request

Go to your fork on GitHub. It will offer a **Compare & pull request** button;
press it. Check the direction carefully before you submit:

```
base repository: Project-MANAS-Research-AI/py-eats   base: main
head repository: <your-username>/py-eats             compare: your branch
```

If the comparison shows hundreds of changed files, the base is wrong.

A description template is filled in for you. Complete it — it is not
decoration:

- **Which issues this addresses.** Write `Closes #1, #2` so they are linked.
- **What you changed and why.** A short paragraph per issue. The reasoning is
  the part worth reading; the diff is already there.
- **Your numbers.** The `py-eats-bench --rules` table, the `py-eats-train` log
  lines, whatever the issue asked for. Say which seed and how many episodes.
  "It improved" is not a number.
- **Issue 2's closing question.** Why the trained network shrugs off a bug
  that destroys the hand-written rule. Nothing tests this; it is read.

Not every **Done when** is checked by a test. The measurements and the
reasoning are read. If they are not in the pull request, they did not happen.

## 10. After you open it

Automated checks start within a minute or so. Your first pull request will
wait for someone to approve the run — that happens to everyone's first
contribution and does not mean anything is wrong.

**Two things get a pull request closed automatically, and both are worth five
seconds of your own time before you push:**

```bash
uv run ruff check --select E9,F63,F7,F82 .   # does it parse?
uv run pytest -m "not challenge" -q          # does the project still hold?
```

Code that does not parse cannot be graded — every check below it fails on
import rather than on its own merits. Code that fails the guard rails has
broken the game rather than improved it, and every number measured after that
means nothing. In both cases you get a comment saying which, and **Reopen**
after fixing re-runs everything.

Everything else reaches a reviewer, **including a pull request that has not
solved anything yet.** Solving one issue out of three is a perfectly good
submission. You are not being filtered on your score.

When the checks finish, open the **Checks** tab and read the summary. You will
get a table like this:

| Issue | Title | Result |
|---|---|---|
| #1 | The dark square is on the wrong end of the snake | solved (4/4 checks) |
| #2 | One of the two hand-written players can no longer survive | not solved (1/4 checks) |
| #3 | Training teaches the snake to die | not solved (0/3 checks) |

**Read that table, not the red or green tick.** The tick also goes red for
reasons that have nothing to do with whether you solved anything. Where a
check failed, the log says exactly what it measured and what it expected —
that message is written to be useful, so read it before asking.

To update your pull request, just push more commits to the same branch. It
updates itself and everything re-runs. Do not close it and open a new one.

If you get review comments, reply to them. Pushing a fix without saying
anything leaves the reviewer guessing whether you understood the point or
changed something at random.

---

## Keeping your fork current

If this repository moves on while you are working:

```bash
git checkout main
git pull upstream main
git push origin main
git checkout your-branch
git rebase main
```

## Things that get a pull request sent back

Closed automatically, before anyone reads it:

- It changes `tests/`, `.github/`, `pyproject.toml`, `uv.lock`,
  `.python-version`, `.gitignore`, `docs/assets/` or
  `src/snake_rl/benchmark.py`.
- It deletes files. Usually this means the branch was cut from the wrong
  place — see "Keeping your fork current" above.
- It changes nothing at all.
- The code does not parse.
- The guard rails fail: the game is no longer reproducible, or no longer
  returns what it advertises.

Sent back by a human, after they have read it:

- It rewrites `src/snake_rl/` and moves no issue forward.
- The base branch is wrong, so the diff is enormous.
- No description, or a description with no numbers in it.
- It claims to solve an issue whose **Done when** it does not meet.

## If you are stuck

Comment on the issue itself. Say which issue, what you tried, and what
happened — including the actual command and the actual output. That gets a
useful answer far faster than "it's not working".
