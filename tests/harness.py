"""Measuring helpers shared by the graded challenge tests.

Everything in here goes through the environment's public surface -- the
constructor, ``reset()``, ``step(action) -> (observation, reward, done,
info)``, ``snake``, ``food``, ``grid_size``, ``observation_dim``,
``action_dim`` -- and works out the rest by *probing* rather than by reading
the source. Nothing imports a private helper and nothing hard-codes a
position in the observation vector that a solution might reasonably have
moved.

That matters because these grade other people's work. A check that assumes
one particular way of fixing an issue fails everybody who fixed it another
way, and the whole point of grading is to tell a working solution from a
broken one, not to tell a familiar solution from an unfamiliar one.
"""

import copy
import inspect
import itertools

import numpy as np

from snake_rl.env import SnakeEnv

# The three actions, in the order the action space uses them.
ACTIONS = (0, 1, 2)  # TURN_LEFT, STRAIGHT, TURN_RIGHT
ACTION_NAMES = ("turn left", "go straight", "turn right")

FREE_SPACE_COUNT = 3

# Observation positions the hand-written rules below read, as documented in
# `SnakeEnv`'s docstring and in the README's "What the snake sees". No issue
# asks for any of these to move, and `tests/test_contract.py` fails loudly if
# one of them stops meaning what it says -- so a solution that shuffled the
# observation gets told that in the guard rails rather than getting a
# confusing failure out of the graded suite.
DANGER_BIT = {0: 2, 1: 0, 2: 1}  # action -> the danger bit that warns about it
FOOD_FORWARD, FOOD_RIGHT = 11, 12

# A solution that would rather just say where it put the free-space block
# than have it measured.
FREE_SPACE_ATTRIBUTES = ("free_space_indices", "free_space_slice")


# --- probing the environment ------------------------------------------------

def step(env, action):
  """Take one step, tolerating either return shape.

  The environment ships returning ``(observation, reward, done, info)``.
  Dropping the ``info`` dict would be an odd thing for a solution to do, but
  accepting a three-value return costs nothing and means a fork that
  simplified the interface is graded on its snake rather than on its tuple.
  """
  result = env.step(action)
  if len(result) not in (3, 4):
    raise AssertionError(
        f"step() returned {len(result)} values; expected (observation, "
        "reward, done) or (observation, reward, done, info)")
  info = result[3] if len(result) == 4 else {}
  return result[0], result[1], result[2], info


def _fresh(env):
  """A throwaway copy of ``env``, for asking "what if" without committing."""
  return copy.deepcopy(env)


def move_is_fatal(env, action):
  """Whether taking ``action`` from this state ends the snake.

  Asked by playing the move on a copy, so it stays true however the
  environment decides what a collision is.
  """
  _, _, _, info = step(_fresh(env), action)
  return bool(info.get("died", False))


def head_after(env, action):
  """Where the head lands if ``action`` is taken, or None if it dies there."""
  probe = _fresh(env)
  _, _, _, info = step(probe, action)
  if info.get("died", False):
    return None
  return probe.snake[0]


def room_after(env, action):
  """Free cells reachable if ``action`` is taken, as a fraction of the room
  the snake needs to survive.

  An independent implementation of the same question the environment's own
  free-space sensor answers, written from public state only. It is used for
  its *ordering* rather than its exact value, so a solution that changes the
  cap or the normalisation still lines up with it.
  """
  start = head_after(env, action)
  if start is None:
    return 0.0

  grid_size = env.grid_size
  # The same tail rule the environment uses: the last cell vacates as the
  # snake moves, so it is not an obstacle.
  body = set(env.snake[:-1])

  def blocked(cell):
    return (not (0 <= cell[0] < grid_size and 0 <= cell[1] < grid_size)
            or cell in body)

  cap = len(env.snake) + 2
  seen, frontier = {start}, [start]
  while frontier and len(seen) < cap:
    row, col = frontier.pop()
    for d_row, d_col in ((-1, 0), (0, 1), (1, 0), (0, -1)):
      cell = (row + d_row, col + d_col)
      if cell not in seen and not blocked(cell):
        seen.add(cell)
        frontier.append(cell)
  return min(len(seen), cap) / cap


def room_truth(env):
  """What the three free-space sensors ought to read, in action order."""
  return np.array([room_after(env, action) for action in ACTIONS])


def sample_states(states=400, seed=0, env_class=SnakeEnv):
  """Observations paired with independently measured free-space truth.

  Random play, restarting on death, so the sample covers open board, tight
  corners and the states just before a crash rather than one tidy trajectory.
  """
  rng = np.random.default_rng(seed)
  env = env_class(seed=seed)
  observation = env.reset()
  observations, truths = [], []
  for _ in range(states):
    observations.append(np.asarray(observation, dtype=float))
    truths.append(room_truth(env))
    observation, _, done, _ = step(env, int(rng.integers(env.action_dim)))
    if done:
      observation = env.reset()
  return np.array(observations), np.array(truths)


# --- finding the free-space block -------------------------------------------
#
# It sits at observation indices 14:17 as shipped, and issue 2 gives nobody a
# reason to move it. Hard-coding that would still be a bet, though, and it is
# a bet that costs nothing to avoid: three *consecutive* channels that track
# the measured room in each of the three directions are the free-space
# sensors, whatever order they arrive in and wherever they sit.

def _declared_free_space_start(env_class):
  for name in FREE_SPACE_ATTRIBUTES:
    value = getattr(env_class, name, None)
    if isinstance(value, slice):
      return value.start
    if isinstance(value, int):
      return value
    if value is not None:
      return int(np.asarray(value).ravel()[0])
  return None


def _match(block, truths, permutation):
  """How well ``block`` read in ``permutation`` order tracks the truth.

  Correlation rather than difference, so a solution that rescaled the sensor
  -- a different flood-fill cap, a raw cell count, a log -- scores on whether
  it moves with the room available, which is the only thing being asked.
  """
  scores = []
  for position, action in enumerate(permutation):
    column, target = block[:, action], truths[:, position]
    if column.std() < 1e-12 or target.std() < 1e-12:
      scores.append(0.0)
    else:
      scores.append(float(np.corrcoef(column, target)[0, 1]))
  return float(np.mean(scores))


def free_space_layout(observations, truths, env_class=SnakeEnv):
  """Locate the free-space block and the order its readings arrive in.

  Returns ``(start, permutation, score, runners_up)`` where ``permutation[i]``
  is the offset within the block holding the sensor for ``ACTIONS[i]`` -- so a
  block at 14 whose readings already line up with the action space reads
  ``(14, (0, 1, 2), ~1.0)``.
  """
  width = observations.shape[1]
  declared = _declared_free_space_start(env_class)
  starts = ([declared] if declared is not None
            else range(width - FREE_SPACE_COUNT + 1))

  ranked = []
  for start in starts:
    block = observations[:, start:start + FREE_SPACE_COUNT]
    if block.shape[1] < FREE_SPACE_COUNT:
      continue
    for permutation in itertools.permutations(range(FREE_SPACE_COUNT)):
      ranked.append((_match(block, truths, permutation), start, permutation))

  ranked.sort(key=lambda row: -row[0])
  score, start, permutation = ranked[0]
  return start, permutation, score, ranked[1:4]


def describe_layout(start, permutation):
  """A plain-English reading of what the sensors are lined up with."""
  return ", ".join(
      f"observation[{start + permutation[position]}] tracks "
      f"{ACTION_NAMES[position]}" for position in range(FREE_SPACE_COUNT))


# --- the two hand-written players -------------------------------------------
#
# Deliberately *not* imported from `snake_rl/benchmark.py`. Issue 2 says in so
# many words that the players are written correctly and are not the thing to
# change, so grading against a submission's own copy of them would let a fix
# that never touched the environment pass -- and would fail an honest
# submission that tidied it. These are the graders' copy, and they read the
# free-space block wherever the probe above found it.

def toward_food(observation):
  """The three actions ordered by how much each closes on the food, best first."""
  score = {1: observation[FOOD_FORWARD],
           2: observation[FOOD_RIGHT],
           0: -observation[FOOD_RIGHT]}
  return sorted(score, key=lambda action: -score[action])


def greedy_food(observation, room):
  """Head for the food, refusing only a move that dies on the very next cell.

  Never reads the free-space block, which is what makes it the control: issue
  2 leaves it at 30.50 whether the sensors are in order or not.
  """
  for action in toward_food(observation):
    if observation[DANGER_BIT[action]] < 0.5:
      return action
  return 1


def food_then_safety(observation, room):
  """Chase the food, but veto a move into measurably less room.

  The free-space reading is capped, so it saturates and ties on most steps,
  and every tie is broken toward the meal -- this picks greedy-food's action
  around 98.6% of the time. The rest is the whole point: the steps where one
  direction is genuinely tighter, which are the steps that would have trapped
  it.
  """
  best = float(np.max(room))
  tied = [action for action in ACTIONS if room[action] >= best - 1e-9]
  for action in toward_food(observation):
    if action in tied:
      return action
  return tied[0]


RULES = {"greedy-food": greedy_food, "food+safety": food_then_safety}


def play_rule(choose, start, seed, episodes, env_class=SnakeEnv):
  """Score a hand-written player over whole episodes. No torch involved.

  Takes the three readings **in the order the environment packed them** --
  ``start`` locates the block, and nothing reorders it. That is the whole
  point: the player is written to expect turn left, straight, turn right, so
  handing it the block as-is is what turns "the sensors are in the wrong
  order" into "this player dies every episode". Un-permuting them here would
  quietly repair the bug before measuring it.

  ``start`` is measured rather than assumed only so that a solution which
  moved the block somewhere else is still graded on its order.
  """
  env = env_class(seed=seed + 10_000)
  lengths, deaths = [], []
  for _ in range(episodes):
    observation, done, info = env.reset(), False, {}
    while not done:
      observation = np.asarray(observation, dtype=float)
      room = observation[start:start + FREE_SPACE_COUNT]
      observation, _, done, info = step(env, choose(observation, room))
    lengths.append(len(env.snake))
    deaths.append(float(info.get("died", False)))
  return float(np.mean(lengths)), float(np.mean(deaths))


def score_rules(start, seeds=(0, 1, 2), episodes=100, env_class=SnakeEnv):
  """Both players over several seeds: ``{name: (mean length, death rate)}``."""
  scored = {}
  for name, choose in RULES.items():
    measured = [play_rule(choose, start, seed, episodes, env_class)
                for seed in seeds]
    scored[name] = (float(np.mean([length for length, _ in measured])),
                    float(np.mean([died for _, died in measured])))
  return scored


def rules_table(scored):
  return "\n".join(f"  {name:<14} mean length {length:6.2f}   died {died:5.1%}"
                   for name, (length, died) in scored.items())


# --- reading the viewer's picture -------------------------------------------

def board_image(viewer):
  """The rendered board, as a ``(grid, grid, 3)`` float array.

  Looks for the shipped ``image_artist`` first and falls back to whatever
  image the board axes is holding, so renaming the attribute does not read as
  a broken viewer.
  """
  artist = getattr(viewer, "image_artist", None)
  if artist is None:
    for axes in viewer.figure.axes:
      if axes.images:
        artist = axes.images[0]
        break
  if artist is None:
    raise AssertionError(
        "the viewer's figure holds no image, so there is no board to check")
  return np.asarray(artist.get_array(), dtype=float)


def draw(viewer):
  """Render the current state, whatever the drawing entry point is called."""
  if callable(getattr(viewer, "_draw", None)):
    viewer._draw()
  else:
    # `update()` steps the environment before drawing, so callers read
    # `viewer.env.snake` back afterwards rather than assuming it held still.
    viewer.update(0)


def staged_snake(viewer, length=14):
  """Put a snake of a known length on the board, head first, and return the env.

  Built by hand rather than played into place: the demo heuristic the viewer
  runs by default ignores danger and rarely survives long enough to grow.

  Laid out row by row rather than in one straight line so the length is not
  capped by the board width, which matters -- the body gradient is spread over
  however many cells there are, so on a *short* snake each step along it is
  large enough to rival the marker, and "which cell is overpainted" stops
  being a question the picture can answer.
  """
  env = viewer.env
  env.reset()
  grid_size = env.grid_size
  last_col = grid_size - 3
  cells, row, col, direction = [], grid_size // 2, last_col, -1
  for _ in range(length):
    cells.append((row, col))
    col += direction
    if col < 0 or col > last_col:
      row += 1
      col = min(max(col, 0), last_col)
      direction = -direction
  env.snake = cells
  # Anywhere off the body: food is drawn too, and a food cell sitting under
  # the snake would corrupt one of the colours being read back.
  env.food = next((row, col) for row in range(grid_size)
                  for col in range(grid_size) if (row, col) not in set(cells))
  return env


def body_colour_steps(image, snake):
  """How far the colour moves between neighbouring cells, head to tail.

  The body is drawn as a smooth gradient with one cell overpainted to mark
  the head, so exactly one of these steps is large and the rest are small.
  *Which* one is large is the whole question in issue 1, and reading it this
  way needs to know neither the marker's colour nor the gradient's.
  """
  colours = np.array([image[row, col] for row, col in snake])
  return np.linalg.norm(np.diff(colours, axis=0), axis=1)


# --- training -------------------------------------------------------------

def train_policy(episodes, seed, **overrides):
  """Train a policy, passing only the arguments this ``train()`` accepts.

  Issue 3 is a one-character fix in the middle of ``train()``, but a
  submission is allowed to have tidied the signature around it, and a
  ``TypeError`` from an unexpected keyword is a terrible way to tell somebody
  their snake does not learn.
  """
  from snake_rl import train as train_module

  wanted = {"episodes": episodes, "seed": seed, "gamma": 0.95,
            "entropy_bonus": 0.03, "checkpoint": None, "plot": None,
            "quiet": True, "grid_size": 12, "horizon": 800, "step_cost": 0.01}
  wanted.update(overrides)
  try:
    accepted = set(inspect.signature(train_module.train).parameters)
  except (TypeError, ValueError):  # pragma: no cover - exotic signature
    accepted = set(wanted)
  return train_module.train(**{key: value for key, value in wanted.items()
                               if key in accepted})


def untrained_policy(seed, hidden_dim=64, env_class=SnakeEnv):
  """A freshly initialised policy on the same seed: the floor to beat.

  The floor is what makes issue 3 checkable without quoting a magic number.
  Training that works clears it; training that pushes the wrong way ends up
  *below* it, which is the state the issue describes.
  """
  import torch

  from snake_rl.models import SnakePolicy

  torch.manual_seed(seed)
  return SnakePolicy(observation_dim=env_class(seed=0).observation_dim,
                     hidden_dim=hidden_dim)


def score_policy(policy, seed, episodes=30, env_kwargs=None):
  """Evaluate with the project's own ``evaluate``, so the numbers are the
  project's numbers rather than a second opinion invented here."""
  from snake_rl.train import evaluate

  return evaluate(policy, seed, episodes,
                  env_kwargs or {"grid_size": 12, "horizon": 800, "step_cost": 0.01})
