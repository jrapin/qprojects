import numpy as np
from . import _players
from ._deck import Card as C
from ._deck import CardList as CL


def test_prepare_learning_output_unplayed():
    value = 12
    output = _players.prepare_learning_output(playable_cards=None, played_card=None, value=value)
    expected_values = [-100] * 32 + [value]
    np.testing.assert_array_equal(output[:, 0], expected_values)
    expected_mask = [1] * 33
    np.testing.assert_array_equal(output[:, 1], expected_mask)
    # errors
    np.testing.assert_raises(TypeError, _players.prepare_learning_output, C("Qh"), None, value)
    np.testing.assert_raises(AssertionError, _players.prepare_learning_output, None, [C("Qh")], value)


def test_prepare_learning_output_played():
    value = 12
    output = _players.prepare_learning_output(playable_cards=CL(["7h", "8h"]), played_card=C("7h"), value=value)
    expected_values = [value] + [-100] * 32
    np.testing.assert_array_equal(output[:, 0], expected_values)
    expected_mask = [1, 0] + [1] * 31
    np.testing.assert_array_equal(output[:, 1], expected_mask)
