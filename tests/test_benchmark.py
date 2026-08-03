"""The benchmark's argument parsing, which is where it can quietly lie.

Not a test that the numbers are right -- that takes half an hour of training.
These cover the two ways a table can come out mislabelled or measuring
something other than what it claims.
"""

import pytest

from snake_rl.benchmark import ENV_DEFAULTS, PRESETS, TRAIN_DEFAULTS, parse_setting


class TestParseSetting:
  def test_values_take_the_type_of_the_default(self):
    """step_cost is subtracted from a reward and grid_size indexes the
    board, so a string would silently misbehave rather than raise."""
    overrides = parse_setting(["grid_size=10", "step_cost=0.02"])
    assert overrides["grid_size"] == 10
    assert isinstance(overrides["grid_size"], int)
    assert overrides["step_cost"] == pytest.approx(0.02)
    assert isinstance(overrides["step_cost"], float)

  def test_training_settings_are_accepted_too(self):
    """One flag covers both tables, so a row can vary the game and the
    optimizer together."""
    overrides = parse_setting(["lr=0.003", "episodes=400"])
    assert overrides["lr"] == pytest.approx(0.003)
    assert overrides["episodes"] == 400
    assert isinstance(overrides["episodes"], int)

  def test_unknown_setting_is_refused(self):
    """A typo that silently did nothing would produce a table that looks
    like a measurement of the thing it was never varying."""
    with pytest.raises(SystemExit):
      parse_setting(["step_cst=2"])

  def test_empty_assignments_give_no_overrides(self):
    assert parse_setting([]) == {}


class TestPresets:
  def test_every_preset_row_is_well_formed(self):
    for name, rows in PRESETS.items():
      for label, overrides in rows:
        assert label, f"{name} has an unlabelled row"
        # An override neither table accepts would be silently ignored, and
        # the row would claim to measure something it never varied.
        assert set(overrides) <= {*ENV_DEFAULTS, *TRAIN_DEFAULTS}, \
            f"{name}: {label}"
