"""Re-measure every comparison this project makes a claim about.

A claim nobody can re-run is a claim nobody can check. Training here is
deterministic per seed (`torch.manual_seed` plus a seeded env), so a given
`--seeds` set reproduces exactly -- which is the only reason quoting a
worst-seed number is meaningful rather than decorative. A different torch
version reproduces a different trajectory, so say which one a table was
measured on before comparing it against another.

  python benchmark.py --preset gamma --rules   # the discount, against the
                                               # bar a trained policy must clear
  python benchmark.py --set lr=0.003 --seeds 0 1 2 3 4   # one ad-hoc row

Always read a learned row against the `--rules` rows rather than against the
untrained floor. The floor is 3.1, and a policy that has learned nothing
useful still clears it easily; the hand-written rules are the number that
tells you whether the learning bought anything.

Output is a markdown table plus the worst..best spread -- on this project the
spread across seeds is usually the more honest number than the mean.
"""

import argparse
import time

import numpy as np
import torch

from snake_rl.env import SnakeEnv
from snake_rl.models import SnakePolicy
from snake_rl.train import evaluate, train

# The env settings every row shares, so the only thing varying between rows
# is the thing the row is about.
ENV_DEFAULTS = {"grid_size": 12, "horizon": 800, "step_cost": 0.01}

# The optimizer's knobs, kept in a second table so a row can say `lr=0.003`
# and `step_cost=0.02` without caring which is which. `episodes` lives here
# too, so a row can buy itself a longer run.
TRAIN_DEFAULTS = {"episodes": 2000, "gamma": 0.95, "lr": 1e-3, "hidden_dim": 64,
                  "entropy_bonus": 0.03, "entropy_final": 0.03}

# A row is (label, overrides). The label is what the table calls it, so a
# preset can name the thing the row varies.
PRESETS = {
  # The shipped configuration on its own, for a plain "is it still what the
  # README says" check.
  "default": [("default", {})],
  # How far ahead an action is held responsible for. Swept wide because the
  # textbook 0.99 sits at the wrong end of the range for a task whose
  # rewards arrive every dozen steps.
  "gamma": [(f"--gamma {discount}", {"gamma": discount})
            for discount in (0.90, 0.93, 0.95, 0.97, 0.99, 0.995)],
  # Whether to let the policy stop exploring once it has learned to eat. A
  # constant weight ships a snake that is still deliberately playing badly
  # some of the time; annealing to zero trades that for the risk of settling
  # into one fixed route. Sweep it and see.
  "entropy": [("constant 0.03 (ships)", {"entropy_final": 0.03}),
              ("0.03 -> 0.01", {"entropy_final": 0.01}),
              ("0.03 -> 0.005", {"entropy_final": 0.005}),
              ("0.03 -> 0.0", {"entropy_final": 0.0})],
  # Where more training stops paying for itself.
  "budget": [(f"--episodes {n}", {"episodes": n})
             for n in (1000, 2000, 3000, 4500, 6000)],
}


def untrained_row(seeds, eval_episodes, quiet):
  """Score a freshly initialised policy: the floor every other row is read against.

  "Length 7.3" carries no information until you know that flailing at random
  scores about 3.1 -- a table without this row lets a config that learned
  nothing pass for a config that learned a little.
  """
  env_kwargs = dict(ENV_DEFAULTS)
  observation_dim = SnakeEnv(seed=0, **env_kwargs).observation_dim
  per_seed = []
  for seed in seeds:
    torch.manual_seed(seed)
    policy = SnakePolicy(observation_dim=observation_dim, hidden_dim=64)
    metrics = evaluate(policy, seed, eval_episodes, env_kwargs)
    per_seed.append(metrics)
    if not quiet:
      print(f"  untrained seed={seed} length={metrics['length']:6.2f}",
            flush=True)
  return ("(untrained)", per_seed)


# Observation indices the hand-written rules below read, matching the layout
# in SnakeEnv._observation.
_DANGER = {0: 2, 1: 0, 2: 1}  # action (left/straight/right) -> its danger bit
_FOOD_FORWARD, _FOOD_RIGHT, _ROOM = 11, 12, slice(14, 17)


class Rule:
  """A hand-written policy wearing SnakePolicy's interface.

  Reads the same observation vector the network reads, and is scored by the
  same `evaluate`, so the comparison is about what was learned rather than
  about who got better inputs.
  """

  def __init__(self, choose):
    self.choose = choose

  def eval(self):
    return self

  def train(self):
    return self

  def sample(self, observation_tensor):
    action = self.choose(observation_tensor.squeeze(0).numpy())
    return torch.tensor(action), None, None


def _toward_food(obs):
  """The 3 actions ordered by how much each closes on the food, best first."""
  score = {1: obs[_FOOD_FORWARD], 2: obs[_FOOD_RIGHT], 0: -obs[_FOOD_RIGHT]}
  return sorted(score, key=lambda action: -score[action])


def greedy_food(obs):
  """Head for the food, refusing only a move that dies on the very next cell."""
  for action in _toward_food(obs):
    if obs[_DANGER[action]] < 0.5:
      return action
  return 1


def food_then_safety(obs):
  """Chase the food, but veto a move into measurably less room.

  Do not read this as "ignores the food and hides". The free-space sensor is
  capped at `len(snake) + 2`, so it saturates: on ~96% of steps some
  direction reads a full 1.0 and on ~47% all three read the same value, and
  every one of those ties is broken toward the meal. Measured over 30k steps
  this picks greedy-food's action **98.6%** of the time.

  The other 1.4% is the whole point. Those are the steps where one direction
  is genuinely tighter than another, and vetoing them is worth 30.50 -> 41.12
  -- because they are exactly the steps that would have trapped it.
  """
  room = obs[_ROOM]
  best = float(np.max(room))
  tied = [action for action in (0, 1, 2) if room[action] >= best - 1e-9]
  for action in _toward_food(obs):  # break ties toward the meal
    if action in tied:
      return action
  return tied[0]


RULES = {"greedy-food": greedy_food, "food+safety": food_then_safety}


def rule_rows(seeds, eval_episodes, quiet):
  """Score every hand-written rule: the bar a trained policy has to clear.

  The untrained row is a floor so low that anything above it reads as
  learning. These rows are the honest comparison -- a policy that does not
  beat `food+safety` has not learned to play Snake, it has learned to be
  worse than two rules a first-year could write.
  """
  measured = []
  for name, choose in RULES.items():
    per_seed = [evaluate(Rule(choose), seed, eval_episodes,
                         dict(ENV_DEFAULTS))
                for seed in seeds]
    if not quiet:
      for seed, metrics in zip(seeds, per_seed):
        print(f"  {name} seed={seed} length={metrics['length']:6.2f}",
              flush=True)
    measured.append((f"{name} (hand-written)", per_seed))
  return measured


def run(rows, seeds, episodes, eval_episodes, quiet):
  measured = []
  for label, overrides in rows:
    # One overrides dict covers both halves of the experiment; which knob a
    # key belongs to is decided by which defaults table it lives in, so a
    # row can vary the game and the optimizer in the same breath.
    env_overrides = {key: value for key, value in overrides.items()
                     if key in ENV_DEFAULTS}
    train_overrides = {key: value for key, value in overrides.items()
                       if key in TRAIN_DEFAULTS}
    env_kwargs = dict(ENV_DEFAULTS, **env_overrides)
    # train() takes the env settings as keyword arguments rather than a
    # dict, so hand it the same values env_kwargs carries for eval.
    settings = dict(TRAIN_DEFAULTS, **train_overrides)
    settings.update(env_kwargs)
    settings["episodes"] = train_overrides.get("episodes", episodes)
    per_seed = []
    for seed in seeds:
      started = time.time()
      policy = train(seed=seed, checkpoint=None, plot=None,
                     quiet=True, **settings)
      metrics = evaluate(policy, seed, eval_episodes, env_kwargs)
      per_seed.append(metrics)
      if not quiet:
        print(f"  {label} seed={seed} length={metrics['length']:6.2f} "
              f"score={metrics['score']:7.1f} "
              f"deaths={metrics['death_rate']:4.2f} "
              f"({time.time() - started:.0f}s)", flush=True)
    measured.append((label, per_seed))
  return measured


def print_table(measured, seeds, threads):
  def mean(per_seed, key):
    return float(np.mean([metrics[key] for metrics in per_seed]))

  print()
  print(f"{len(seeds)} seeds ({', '.join(str(s) for s in seeds)}), "
        f"{threads} torch threads, "
        f"mean final length over the sampled eval episodes.")
  print()
  print("| config | mean length | worst..best | score | died | steps/food |")
  print("|---|---|---|---|---|---|")
  for label, per_seed in measured:
    lengths = [metrics["length"] for metrics in per_seed]
    print(f"| {label} | {np.mean(lengths):.2f} | "
          f"{min(lengths):.2f} .. {max(lengths):.2f} | "
          f"{mean(per_seed, 'score'):.1f} | "
          f"{mean(per_seed, 'death_rate'):.0%} | "
          f"{mean(per_seed, 'steps_per_food'):.1f} |")


def parse_setting(assignments):
  """`--set step_cost=0.02 lr=0.003` -> an overrides dict.

  Values are parsed against the type of the existing default, so
  `episodes=400` stays an int and `step_cost=0.02` becomes a float -- a
  string would silently do the wrong thing in the arithmetic downstream.
  """
  known = {**ENV_DEFAULTS, **TRAIN_DEFAULTS}
  overrides = {}
  for assignment in assignments:
    key, _, raw = assignment.partition("=")
    if key not in known:
      raise SystemExit(f"unknown setting {key!r}; "
                       f"choose from {', '.join(sorted(known))}")
    default = known[key]
    overrides[key] = type(default)(raw)
  return overrides


def main():
  """The `py-eats-bench` command. See `train.main` on why this is a function."""
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--preset", choices=sorted(PRESETS),
                      help="a named set of rows to compare")
  parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                      dest="settings",
                      help="any env or training setting, measured as a single "
                           "row against the defaults, e.g. lr=0.003")
  parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                      help="training seeds; more seeds, tighter spread")
  parser.add_argument("--episodes", type=int, default=2000)
  parser.add_argument("--eval-episodes", type=int, default=100,
                      help="sampled episodes each trained policy is scored on")
  parser.add_argument("--threads", type=int, default=2,
                      help="torch intra-op threads, pinned because the count "
                           "changes the answer: the same seed at a different "
                           "thread count reorders float reductions and "
                           "diverges over 1500 episodes (about half a length "
                           "on this task). Pinned, runs reproduce exactly; "
                           "change it and expect a small shift in every row")
  parser.add_argument("--untrained", action="store_true",
                      help="prepend an untrained-policy row as the floor the "
                           "trained rows should be read against")
  parser.add_argument("--rules", action="store_true",
                      help="prepend the hand-written policies -- the bar a "
                           "trained policy actually has to clear, as opposed "
                           "to the untrained floor, which it clears trivially")
  parser.add_argument("--quiet", action="store_true",
                      help="only print the final table")
  args = parser.parse_args()

  if args.settings:
    if args.preset:
      raise SystemExit("--set builds its own row; a --preset already sets the "
                       "values each of its rows is about")
    overrides = parse_setting(args.settings)
    # Name the overrides in the label: a table row that says "default" when it
    # was measured under lr=0.003 is worse than no table, because it reads as
    # the shipped configuration.
    label = " ".join(f"{key}={value}" for key, value in overrides.items())
    rows = [(label, overrides)]
  else:
    rows = PRESETS[args.preset or "default"]

  # Before any training: this is what makes the tables reproducible at all.
  torch.set_num_threads(args.threads)

  started = time.time()
  measured = []
  if args.untrained:
    measured.append(untrained_row(args.seeds, args.eval_episodes,
                                  args.quiet))
  if args.rules:
    measured += rule_rows(args.seeds, args.eval_episodes, args.quiet)
  measured += run(rows, args.seeds, args.episodes, args.eval_episodes,
                  args.quiet)
  print_table(measured, args.seeds, args.threads)
  print(f"\nTotal {time.time() - started:.0f}s for "
        f"{len(rows) * len(args.seeds)} runs.")


if __name__ == "__main__":
  main()
