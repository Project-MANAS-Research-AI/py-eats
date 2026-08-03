# Automation

Maintainer notes. All free/built-in GitHub Actions, no paid add-ons, no secrets
beyond `GITHUB_TOKEN` (one optional PAT — see below).

The same spine as the sibling `toy-quadruped` challenge — `pr-policy`,
`pr-checks`, `tests`, and the two labelers — around the thing that one does not
have: a graded suite on a separate branch. That brings `pr-triage.yml` to act on
the result and `grade-fork.yml` to run the same grading without a pull request,
plus the housekeeping below.

**The question the automation answers is not "is this correct?"** It is *does
this branch run, and is it worth a reviewer's time?* A person answers the rest.

## What reaches a human, and what does not

This is the whole design in one table.

| What arrives | What happens | Reaches a reviewer |
|---|---|---|
| Empty PR, or one editing the tests / CI / lockfile / `docs/assets/` / `benchmark.py` | commented on and **closed** by `pr-policy.yml` | no |
| A PR that deletes files | commented on and **closed** by `pr-policy.yml` | no |
| Code that does not parse | commented on and **closed** by `pr-triage.yml` | no |
| Code that fails the guard rails | commented on and **closed** by `pr-triage.yml` | no |
| Anything else — including a submission that solved *nothing* | labelled `ready-for-review`, assigned, scorecard posted | **yes** |

The last row is the one to keep in mind when changing any of this. A branch
that solves one issue of three, or none, is honest work and goes to review with
its scorecard attached. Only a branch that does not **run** is turned away — and
even then it is turned away with a comment saying exactly what broke, how to
reproduce it in five seconds locally, and an invitation to fix and reopen.
Nothing is deleted and nothing is judged; a closed PR reopens with one click and
re-runs from the top.

Set the **`MAINTAINER` repository variable** to a GitHub username (Settings →
Secrets and variables → Actions → Variables) and everything that survives triage
is assigned to them. Unset, the `ready-for-review` label is still applied and is
what the review queue filters on.

### Checks — these can close a pull request

| Workflow | Trigger | What it does |
|---|---|---|
| `pr-policy.yml` | PR opened/updated | **The first gate, on filenames alone.** Closes an empty PR, one that edits the files deciding its own grade, one that edits `benchmark.py`, and one that deletes anything. Runs on `pull_request_target`, so a PR cannot switch it off. |
| `pr-checks.yml` | PR to `main` | **The grader.** Restores this repo's tests over the PR's, checks it parses, runs the guard rails, requires a real change, posts a per-issue scorecard, fails on a regression, and renders the board so a reviewer can watch it. Writes a verdict artifact. |
| `pr-triage.yml` | after `pr-checks` | Reads that verdict. Closes a branch that does not parse or does not hold together; routes everything else to a human. |
| `tests.yml` | push/PR to `main` | Syntax and undefined names (blocking), style (advisory), guard rails. Keeps the default branch honest. |
| `grade-fork.yml` | manual, Actions tab | The same grading, against a fork with no PR open. |
| `codeql.yml` | push/PR, weekly | Static analysis. **Skipped while the repo is private** — see below. |
| `dependency-review.yml` | PR to `main` | Flags vulnerable dependencies. **Skipped while the repo is private** — see below. |

### Housekeeping — these only sort and greet

| Workflow | Trigger | What it does |
|---|---|---|
| `pr-labeler.yml` + its config | PR opened/updated | Labels by changed path (`env/rules`, `policy`, `training`, `viewer`, `benchmark`, `tests`, `docs`, `dependencies`, `ci`). |
| `community.yml` + `labeler.yml` config | issue/PR opened | Labels by keyword in title/body, posts a comment explaining them. |
| `greetings.yml` | first issue/PR from a user | One-time welcome. |
| `stale.yml` | daily | Marks 60-day-quiet items stale, closes after 14 more. `submission` is exempt — one waiting on review is not abandoned. |

There is deliberately **no Dependabot config**, the same as the sibling
toy-quadruped challenge. It opens a batch of version-bump pull requests the
moment the repo exists and a fresh batch every week, and a closed pull request
is permanent — a challenge tracker whose pull request list is mostly bot noise
buries the submissions it exists to hold. The exemptions for `dependabot` that
remain in `pr-policy.yml` and `pr-checks.yml` cost nothing and mean the checks
behave sensibly if it is ever switched back on.

**None of the housekeeping workflows check correctness.** A labeler matches
filenames and keywords, so it labels a broken fix and a correct one identically.

## What `pr-checks.yml` verifies, in order

Cheapest first, and the order is load-bearing: the first two decide whether a
human ever sees the PR at all.

1. **Does it parse.** `ruff --select E9,F63,F7,F82`. Five seconds. A branch with
   a syntax error or an undefined name cannot be graded — every test below it
   fails on import rather than on its own merits, which tells a reviewer nothing
   about the work.
2. **Does the project still hold together.** The guard-rail suite: finite
   observations of the advertised size, a real `done` flag, junk actions
   survived, the horizon respected, the same seed still producing the same
   trajectory. Breaking one of these has broken the environment rather than
   improved it, and every measurement taken afterwards is meaningless.
3. **Is there a real change.** At least one changed line under `src/snake_rl/`,
   ignoring whitespace, blank lines and comment-only lines, so reformatting or
   adding a docstring does not read as work.
4. **What does it fix.** The graded suite, then the *same* suite against the
   base branch, so the result reads as a change rather than an absolute.
5. **Nothing regressed, and graded code moved something.** See below.
6. **The board renders.** Two GIFs, checked for a plausible size — a viewer that
   drew nothing still writes a valid GIF header. Uploaded alongside the
   committed originals so a reviewer can open them side by side.

Steps 1 and 2 are the only ones that feed the `broken` field in the verdict.
Everything after is scoring, and a low score is a perfectly good reason to still
be reviewed.

## What a pull request cannot do

Five separate holes, five separate plugs. Only the first is obvious.

**1. Delete the workflow that would have graded it.** For a `pull_request` event
GitHub reads the workflow files *from the PR's own merge ref*. Delete
`.github/workflows/pr-checks.yml` in the branch and it simply never runs — and a
check that never ran is not a check that failed, so the PR sits there green.
`pr-policy.yml` triggers on **`pull_request_target`**, which reads the workflow
file from the base branch instead. It is the one file a PR cannot reach.

> `pull_request_target` runs with a writable token in this repo's context, so
> `pr-policy.yml` must never check out or run the PR's code. It reads the
> changed-*filenames* list from the API and nothing else. `pr-triage.yml` is
> under the same rule: it downloads one small JSON artifact and quotes every
> value it reads into a shell variable. Keep both that way.

**2. Rewrite the tests, or the environment they run in.** `pr-policy.yml`
rejects any PR touching `.github/**`, `tests/**`, `pyproject.toml`, `uv.lock`,
`.python-version`, `.gitignore` or `docs/assets/**`. The environment files are on
that list for the same reason the tests are — the lock and the interpreter pin
decide what the run actually executes on, so everyone is checked on the same one.
`docs/assets/` is there because it is the answer key: editing the committed GIF
is not a fix. Dependabot is exempt (bumps are its entire job); a maintainer
changing CI on purpose labels the PR `ci-change`. Belt and braces: `pr-checks.yml`
restores `tests/` and `.github/scripts/` from the `tests` branch and the
environment files from the base branch anyway.

**3. Delete the project out from under the checks.** Any deletion is rejected,
not just deletions of the project's own modules. Nothing in the three issues is
solved by removing a file, so a deletion is either an accident — a bad merge, a
stray `git rm`, a branch cut from the wrong place — or an attempt to make a
failing check disappear by removing what it reads. Removing `env.py` in
particular makes the suite error on import rather than fail, which reads very
differently in a log.

**4. Pass by changing nothing.** An empty PR is closed by `pr-policy.yml`, and
one whose only changes are cosmetic fails step 3 above. Separately: the challenge
tests are meant to fail on the shipped repo, so running them advisory-only would
be green for a PR that fixed nothing *and* for one that broke something. The run
grades **the PR and the base branch with the same tests** and fails if any issue
regressed.

**5. Pass by moving nothing.** Rule 4 only asks about regressions, so a PR that
renamed a variable at random — attempting no issue, breaking no issue — would go
green with every graded test still failing underneath it. A PR whose diff touches
`src/snake_rl/{env,models,train,viewer}.py` must move at least one issue's
scorecard forward. Passing more checks is enough; the issue does not have to be
finished, so an honest half-solution still reads as progress. Docs, CI and
Dependabot PRs touch no graded code and are held to nothing here.

`tests/conftest.py` writes the scorecard as JSON when `CHALLENGE_SCORECARD_JSON`
is set — `{"2": {"passed": 1, "total": 4}, ...}`, counts rather than a bare
pass/fail, so partial progress is visible;
`.github/scripts/compare_scorecards.py` diffs two of them.

### `benchmark.py` is protected, and that is a design decision

Issue #2's symptom is a hand-written player that dies every episode. You can make
that symptom disappear by editing the player until it copes with the numbers it
is handed — which leaves the sensors backwards, the trained network still
misled, and the issue unfixed. `pr-policy.yml` rejects the shortcut by name, with
the reason spelled out.

The graded suite backs this up independently: `tests/harness.py` carries the
grader's **own** copy of both players rather than importing them from
`benchmark.py`, so a submission's edits to that file cannot affect its score in
either direction.

## The two checks that need Advanced Security

`codeql.yml` and `dependency-review.yml` are free and automatic on a public
repo. On a **private** one both need GitHub Advanced Security enabled under
Settings → Code security, and without it they fail on every single run — CodeQL
on a permissions error, dependency-review on the missing dependency graph.

Both are therefore gated on `if: ${{ !github.event.repository.private }}` and
**skip** rather than fail. A check that is permanently red teaches everyone to
ignore red, and this repo spends `CONTRIBUTING.md` and the greeting message
telling candidates to read the scorecard rather than the tick.

## Dependencies

`pyproject.toml` is the single source: dependencies, pytest configuration and
the build all live there, and the project is a real installable package under
`src/`, so nothing has to put the repo root on `sys.path`.

**Everyone runs one command: `uv sync`.** It creates the environment, installs
the locked versions and installs the project. `torch` and `matplotlib` sit in
`agent` and `viewer` dependency groups that `tool.uv.default-groups` turns on, so
a plain sync gets the lot.

Unlike the sibling toy-quadruped challenge, **no job here can run lean.** That
one's `tests.yml` opts out of torch with `--no-default-groups --group dev`
because its contract suite is pure numpy; this repo's guard rails cover the
trainer and the viewer as well, and issue #1 is graded off a rendered board while
issue #3 is graded by actually training. `--locked` everywhere refuses to run
against a `uv.lock` that has drifted from `pyproject.toml`.

**The PyTorch CPU wheel index lives in `pyproject.toml` under
`[[tool.uv.index]]` with `explicit = true`.** Not in a requirements file: an
`--extra-index-url https://download.pytorch.org/whl/cpu` line there makes
Dependabot resolve torch to the local version `2.13.0+cpu` and open a PR pinning
`torch>=2.13.0+cpu` — which pip refuses outright, because a local version label
only works with `==` or `!=`, breaking every check that installs anything.
`explicit = true` also means nothing is drawn from that index unless routed
there by name, so it cannot start shadowing other packages.

Grading costs about 50 seconds of test time on top of the install: roughly 7s
for the two hand-written players, 28s for the five-seed 300-episode training
comparison, and 17s for the 600-episode growth run.

## Submissions are pull requests

A candidate forks the repo, does the work on their fork, and opens one pull
request here with everything they fixed. Dependabot and maintainer work come
through the same door and get the same treatment.

Three things about fork PRs that will bite before anything else does:

1. **The first run needs a click.** GitHub holds workflow runs from first-time
   contributors until a maintainer presses *Approve and run*, and every candidate
   is a first-time contributor. Nothing is broken; nobody has approved it yet.
   Note the consequence for triage: until that click, `pr-checks.yml` has not
   run, so nothing has been closed and nothing has been routed — only
   `pr-policy.yml` has spoken, because `pull_request_target` runs in this repo's
   context and does not wait for approval.
2. **A private repo cannot be forked by someone who cannot see it.** If
   candidates are outside the org, either add them or make the repo public.
3. **Every submission is readable by every other candidate.** A PR diff is
   visible to anyone with repo access, so the first correct submission is visible
   to everyone still working. That is inherent to grading by PR, not something
   the automation can fix. If it matters, review and merge or close quickly, or
   keep submissions in candidates' own forks and use `grade-fork.yml` instead.

## Two branches

| Branch | Holds | Who it is for |
|---|---|---|
| `main` (default) | baseline code with the three bugs, the guard rails, `tests/harness.py` | candidates — no answers |
| `tests` | the same **plus** `tests/test_challenge.py` and `tests/conftest.py` | you — the graded suite and the scorecard |

The `tests` branch carries the *complete* suite, not just the challenge half.
That matters: the overlay replaces the fork's whole `tests/` directory, so if the
guard rails were not on that branch too, a fork could weaken them and the grader
would never notice.

Candidates running `pytest` on their fork see the guard rails and no scorecard.
They are not being graded in public, and they cannot read their own result.

`tests/harness.py` is deliberately on **both** branches. It is what drives the
game during grading, and `tests/test_contract.py` imports it — a guard rail that
cannot import is not a guard rail. It contains no answers: it measures the
free-space sensors rather than asserting anything about them.

Keeping the two in step is one command, run from the repo root:

```bash
git checkout tests && git merge --ff-only main && git push origin tests
```

If that refuses to fast-forward, `main` has been rebased and the `tests` branch
needs rebuilding on top of it — the graded files are the only two that differ.

## The one rule

**Code comes from the submission, tests come from the `tests` branch.** The
grading workflow deletes the submission's `tests/` and restores that copy before
running anything. A fork that rewrites the tests to pass gets graded on the real
ones.

**Be clear about what this does and does not buy.** It makes the tests
*tamper-proof*: nothing a candidate does to their fork changes what grades it. It
does **not** make them secret. A fork copies every branch, and anyone who can see
this repo can read `tests`. That is fine as designed — each test asserts exactly
the "Done when" its issue already states in public, so reading it reveals
nothing, and passing it still requires doing the work.

If you ever want the grading genuinely unreadable, a branch cannot do it — the
suite has to move to a separate repo the candidates have no access to, with
`grade-fork.yml` living there too.

## Grading a shared fork

Actions tab → **Grade a fork** → Run workflow, then fill in the fork as
`owner/repo`. Or:

```bash
gh workflow run grade-fork.yml -R Project-MANAS-Research-AI/py-eats \
  -f fork=student/py-eats
```

The job summary gets the scorecard plus the exact commit that was graded. The run
fails if anything is unsolved — that is expected and is just a red/green signal,
not a verdict.

## Private-repo caveat

If this repo is private, forks of it are private too, and `actions/checkout`
cannot read them with the default job token. Add a PAT with `repo` read scope as
the `GRADER_TOKEN` secret; the workflow falls back to `github.token` when it is
absent, which is enough only if the fork is public.

## Where the grading detail lives

Which check corresponds to which fix, and where every threshold came from, is in
`docs/grading.md` on the **`tests` branch** — next to the suite it describes, and
off the branch candidates read.
