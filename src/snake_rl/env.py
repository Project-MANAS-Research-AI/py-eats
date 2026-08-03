"""A tiny grid-based Snake environment for discrete-action RL."""

import numpy as np

# Action / heading index -> (row, col) step. Clockwise order (UP, RIGHT,
# DOWN, LEFT) so "turn right" is +1 and "turn left" is -1, mod 4.
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))

# Actions are *relative* to the current heading: the turn each one applies
# to the heading index, mod 4.
TURN_LEFT, STRAIGHT, TURN_RIGHT = 0, 1, 2
_TURNS = (-1, 0, 1)


class SnakeEnv:
  """Classic Snake on a small square grid.

  Observation (18 values; see `_observation` for how each block is built):
    danger_straight, danger_right, danger_left -- relative to current
      heading, whether the adjacent cell in that direction is a wall or the
      snake's own body.
    moving_up, moving_right, moving_down, moving_left -- one-hot heading.
    food_up, food_down, food_left, food_right -- absolute sign comparison of
      the food cell against the head cell (not mutually exclusive: diagonal
      food sets two bits).
    food_forward, food_right, food_distance -- the food in the snake's own
      frame, which the four absolute bits above cannot express.
    free_left, free_straight, free_right -- how much room each move leads
      into, in the same order as the action space.

    food_freshness -- how much of the starvation clock is left, in [0, 1].
      This is the survival half of the observation: running the clock out
      ends the episode and costs the same as a crash, and nothing else the
      snake can see says how close that is. A deadline the policy cannot
      perceive is not a harder task, just a noisier one. (The *score* bonus
      is a separate quantity -- see `route_efficiency` -- and needs no input
      of its own, because walking straight at food the snake can already
      see is what earns it.)

  Action: one of 3 moves *relative* to the current heading -- TURN_LEFT,
    STRAIGHT, TURN_RIGHT. No single turn rotates the heading by 180
    degrees, so reversing into the snake's own neck is impossible. This
    also puts actions in the same frame as the danger bits above, making
    "danger_right is set, so don't turn right" directly learnable.

  Reward: split into two named terms, so a caller can report how much of a
    return came from eating and how much from staying alive.

    The *food* term is everything about getting the meal: the meal's own
    points for eating it, a small symmetric +/-0.1 for moving closer to /
    farther from the food, and `-step_cost` for every non-eating step. The
    meal's points are exactly what the scoreboard records for it, with no
    exceptions. The shaping is what gives REINFORCE a signal
    before the snake ever eats by luck; it is small next to the meal and
    cancels out if the snake just paces back and forth. It nets exactly zero
    around a closed loop, so without the standing step cost a snake that
    finds a safe cycle is never punished for riding it.

    The *survival* term is a flat -10 for not surviving -- hitting a wall,
    hitting its own body, or running the starvation clock out. All three are
    the same event as far as the snake is concerned, and pricing them
    differently is how you teach it to prefer one kind of dying.

  Episodes end on death, on filling the board, after `starvation_limit`
    steps without eating, or at the horizon.

  Score: a classic arcade Snake score, 10 for the apple plus up to 10 more
    for having walked a direct route to it -- see `route_efficiency`. This is
    also exactly what the reward pays for a meal, so the route the snake is
    graded on is the route it is trained for. A flat +10 told the policy that
    a meal reached in two steps and one reached in forty were worth the same.
  """

  observation_dim = 18
  action_dim = 3

  def __init__(self, grid_size=12, horizon=800, seed=0, step_cost=0.01):
    self.grid_size = grid_size
    self.horizon = horizon
    self.step_cost = step_cost
    # Max Manhattan distance across the board is 2 * (grid_size - 1), so
    # this leaves slack to reach any food while staying well under horizon.
    self.starvation_limit = 4 * grid_size
    self.rng = np.random.default_rng(seed)
    self.reset()

  def reset(self):
    self.step_count = 0
    self.steps_since_food = 0
    self.score = 0
    self.foods_eaten = 0
    center = self.grid_size // 2
    self.snake = [(center, center), (center, center - 1), (center, center - 2)]
    self.direction = RIGHT
    self.food = self._place_food()
    return self._observation()

  def step(self, action):
    self.step_count += 1
    self.direction = (self.direction + _TURNS[int(action)]) % 4
    d_row, d_col = _DIRECTIONS[self.direction]
    head_row, head_col = self.snake[0]
    new_head = (head_row + d_row, head_col + d_col)

    previous_distance = self._food_distance()

    hit_wall = not (0 <= new_head[0] < self.grid_size and
                    0 <= new_head[1] < self.grid_size)
    ate_food = (not hit_wall) and new_head == self.food
    # The tail cell vacates this step unless the snake is growing, so it
    # isn't a collision to move into it.
    body = self.snake if ate_food else self.snake[:-1]
    hit_self = (not hit_wall) and new_head in body
    died = hit_wall or hit_self

    filled_board = False
    points = 0
    if not died:
      self.snake.insert(0, new_head)
      if ate_food:
        # Scored before steps_since_food is cleared: the bonus is paid on how
        # fresh the food was when it was reached.
        points = self._score_food()
        self.score += points
        self.foods_eaten += 1
        self.steps_since_food = 0
        filled_board = len(self.snake) >= self.grid_size * self.grid_size
        # A full board has nowhere to put food, and _place_food() would
        # divide by an empty cell list.
        if not filled_board:
          self.food = self._place_food()
      else:
        self.snake.pop()
        self.steps_since_food += 1

    starved = self.steps_since_food >= self.starvation_limit
    done = died or filled_board or starved or self.step_count >= self.horizon
    reward, food_reward, survival_reward = self._reward(
      ate_food, died, previous_distance, starved, points)

    info = {"ate": ate_food, "died": died, "starved": starved,
            "filled_board": filled_board,
            "score": self.score, "points": points,
            "length": len(self.snake), "food_reward": food_reward,
            "survival_reward": survival_reward}
    return self._observation(), reward, done, info

  def _reward(self, ate_food, died, previous_distance,
              starved=False, points=0):
    """The two reward axes, and their sum.

    Returns `(reward, food_reward, survival_reward)` -- the components come
    back so callers can report how much of a return came from eating versus
    from staying alive, which is the whole reason for splitting them.

    The food axis gets a symmetric +/-0.1 for closing on or losing ground to
    the meal. It is a difference, so it telescopes: around any closed loop it
    sums to zero and cannot be farmed by pacing back and forth. The standing
    `step_cost` is what stops riding a safe loop from being free.
    """
    food = survival = 0.0
    # Starving is dying: both ways of not surviving cost the same, so the
    # clock is a real deadline rather than a free exit.
    if died or starved:
      survival = -10.0
    elif ate_food:
      # Pay exactly what the scoreboard pays, filling the board included, so
      # the policy is never graded on a speed it was not trained for.
      food = float(points)
    else:
      # Small, symmetric nudge toward the food, minus the standing cost of
      # having taken a step at all.
      new_distance = self._food_distance()
      if new_distance < previous_distance:
        food = 0.1
      elif new_distance > previous_distance:
        food = -0.1
      food -= self.step_cost
    return food + survival, food, survival

  def food_freshness(self):
    """Fraction of the starvation clock still left, in [0, 1].

    The snake is always racing this clock -- it is what ends the episode if
    it dawdles -- so it is the natural deadline to both pay the speed bonus
    against and show the policy.
    """
    return max(0.0, 1.0 - self.steps_since_food / self.starvation_limit)

  @property
  def food(self):
    return self._food

  @food.setter
  def food(self, cell):
    """Placing food also records how far away it was.

    A property rather than a line inside `_place_food` so the yardstick can
    never go stale: tests and demos set `env.food` directly to build a
    situation, and any of those that forgot to update the distance by hand
    would silently score the next meal against the wrong trip.
    """
    self._food = cell
    head_row, head_col = self.snake[0]
    # max(1,...): food never spawns on the head, but a direct assignment
    # could, and this is a divisor.
    self.food_spawn_distance = max(1, abs(cell[0] - head_row)
                                   + abs(cell[1] - head_col))

  def route_efficiency(self):
    """How direct the route to the current food has been, in [0, 1].

    `1.0` means the snake walked straight at it: the food was `d` cells away
    when it appeared and the snake is eating it on step `d`. Half means it
    took twice the necessary steps.

    Measuring against the food's own spawn distance is what makes this fair.
    The obvious alternative -- how much of the starvation clock is left --
    sounds equivalent and is not: the clock is far longer than a typical
    trip, so nearly every policy scores the same against it and the bonus
    discriminates nothing. A near meal and a far one are also not the same
    task, and a fixed clock prices them as though they were.
    """
    return self.food_spawn_distance / max(self.steps_since_food,
                                          self.food_spawn_distance)

  def _score_food(self):
    """Points for the meal the snake is about to eat.

    Arcade Snake pays you for the apple and pays you more for not dawdling,
    so this is a flat 10 plus up to the same again for taking a direct route.

    This *is* what the reward pays for a meal, so the speed the snake is
    graded on is the speed it is trained for.
    """
    return round(10 * (1.0 + self.route_efficiency()))

  def _food_distance(self):
    """Manhattan distance to the food.

    It moves by exactly one cell per step, which is what lets the +/-0.1
    shaping telescope to zero around a closed loop. A BFS distance through
    open cells sounds better and is not: the snake's own body opens and
    closes shortcuts, so that distance jumps by +3 on 2.5% of steps and never
    by -3, and one +3 step penalised once at -0.1 against three +0.1 steps
    walking it back nets +0.2 for a detour that ended where it started.
    """
    head_row, head_col = self.snake[0]
    food_row, food_col = self.food
    return abs(food_row - head_row) + abs(food_col - head_col)

  def _place_food(self):
    occupied = set(self.snake)
    empty_cells = [(row, col) for row in range(self.grid_size)
                   for col in range(self.grid_size)
                   if (row, col) not in occupied]
    if not empty_cells:
      # Only reachable on a full board, which ends the episode anyway; keep
      # the food where it is rather than indexing an empty list.
      return self.food
    return empty_cells[self.rng.integers(len(empty_cells))]

  def _blocked(self, cell):
    """Whether `cell` is off the board or occupied by the body."""
    if not (0 <= cell[0] < self.grid_size and 0 <= cell[1] < self.grid_size):
      return True
    # Same tail rule as step(): the last cell vacates as the snake moves.
    return cell in self._body_without_tail

  def _free_space(self, direction):
    """Reachable free area if the snake steps `direction`, as a fraction of
    the room it needs to survive.

    A snake of length L is trapped the moment it enters a pocket smaller
    than L, and the one-cell danger bits cannot see that at all -- the
    entrance to a fatal pocket looks exactly like open board. Counting is
    capped at L + 2, so this answers "is there room to survive here" rather
    than measuring the whole board, and stays cheap while the snake is
    short.
    """
    start = self.cell_ahead(direction)
    if self._blocked(start):
      return 0.0
    return self._flood(start, len(self.snake) + 2)

  def _flood(self, start, cap):
    """Free cells reachable from `start`, as a fraction of `cap`.

    Capped rather than exhaustive, which also makes the answer independent
    of the order cells come off the frontier: either the fill reaches `cap`
    and reports 1.0, or it exhausts a smaller pocket and reports its true
    size.
    """
    seen = {start}
    frontier = [start]
    while frontier and len(seen) < cap:
      row, col = frontier.pop()
      for d_row, d_col in _DIRECTIONS:
        cell = (row + d_row, col + d_col)
        if cell not in seen and not self._blocked(cell):
          seen.add(cell)
          frontier.append(cell)
    # min(): the frontier adds up to four neighbours per pop, so `seen` can
    # overshoot the cap on the last expansion and report more than 1.0.
    return min(len(seen), cap) / cap

  def cell_ahead(self, direction):
    """The cell one step along `direction` from the head."""
    d_row, d_col = _DIRECTIONS[direction]
    head_row, head_col = self.snake[0]
    return (head_row + d_row, head_col + d_col)

  def _danger(self, direction):
    d_row, d_col = _DIRECTIONS[direction]
    head_row, head_col = self.snake[0]
    cell = (head_row + d_row, head_col + d_col)
    if not (0 <= cell[0] < self.grid_size and 0 <= cell[1] < self.grid_size):
      return 1.0
    return 1.0 if cell in self.snake[:-1] else 0.0

  def _observation(self):
    # Cached once per observation: the free-space sensors probe
    # the body thousands of times per step, and re-slicing snake[:-1] on
    # each probe is what makes that expensive.
    self._body_without_tail = set(self.snake[:-1])
    danger_straight = self._danger(self.direction)
    danger_right = self._danger((self.direction + 1) % 4)
    danger_left = self._danger((self.direction - 1) % 4)
    heading = [1.0 if self.direction == index else 0.0 for index in range(4)]

    head_row, head_col = self.snake[0]
    food_row, food_col = self.food
    food_direction = [1.0 if food_row < head_row else 0.0,
                      1.0 if food_row > head_row else 0.0,
                      1.0 if food_col < head_col else 0.0,
                      1.0 if food_col > head_col else 0.0]

    # The food in the snake's *own* frame. The four bits above are absolute
    # (north/south/east/west) while the danger bits and the actions are
    # heading-relative, so on their own the network has to learn a heading x
    # food product just to answer "is the food to my left" -- a
    # multiplicative interaction a small tanh MLP burns a lot of capacity
    # on. Projecting the food offset onto the heading and its right-hand
    # normal states that directly, and signed values carry distance too,
    # which the pure sign bits cannot express at all.
    forward_row, forward_col = _DIRECTIONS[self.direction]
    right_row, right_col = _DIRECTIONS[(self.direction + 1) % 4]
    offset_row, offset_col = food_row - head_row, food_col - head_col
    food_forward = (offset_row * forward_row + offset_col * forward_col)
    food_right = (offset_row * right_row + offset_col * right_col)
    # Normalized to roughly [-1, 1] / [0, 1] so every input stays on the
    # same scale as the existing bits.
    food_distance = (abs(offset_row) + abs(offset_col)) / (2.0 * self.grid_size)
    # The three moves, in the same TURN_LEFT / STRAIGHT / TURN_RIGHT order
    # the action space uses, so each free-space sensor lines up with the
    # action it informs.
    directions = [(self.direction + turn) % 4 for turn in (1, 0, -1)]
    free_space = [self._free_space(direction) for direction in directions]

    # The food clock goes last, so the block above stays in the order the
    # class docstring lists it.
    return np.array([danger_straight, danger_right, danger_left,
                     *heading, *food_direction,
                     food_forward / self.grid_size,
                     food_right / self.grid_size,
                     food_distance,
                     *free_space,
                     self.food_freshness()], dtype=np.float32)
