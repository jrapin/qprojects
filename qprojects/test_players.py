from unittest import TestCase
import genty
import numpy as np
import keras
from keras import backend as K
from . import _players
from . import _game
from ._deck import Card as C
from ._deck import CardList as CL


def test_prepare_learning_output_unplayed():
    value = 12
    output = _players.prepare_learning_output(playable_cards=None, played_card=None, value=value, value_weight=5)
    expected_values = [-100] * 32 + [value]
    np.testing.assert_array_equal(output[:, 0], expected_values)
    expected_mask = [1] * 32 + [1.25]
    np.testing.assert_array_equal(output[:, 1], expected_mask)
    # errors
    np.testing.assert_raises(TypeError, _players.prepare_learning_output, C("Qh"), None, value)
    np.testing.assert_raises(AssertionError, _players.prepare_learning_output, None, [C("Qh")], value)


def test_prepare_learning_output_played():
    value = 12
    output = _players.prepare_learning_output(playable_cards=CL(["7h", "8h"]), played_card=C("7h"), value=value, value_weight=5)
    expected_values = [value] + [-100] * 32
    np.testing.assert_array_equal(output[:, 0], expected_values)
    expected_mask = [5, 0] + [1] * 31
    np.testing.assert_array_equal(output[:, 1], expected_mask)


def test_masked_mean_squared_error():
    y_true = K.variable(np.array([[0.3, 0.2, 3.1], [0, 0, 1]]).T[None, :, :])
    np.testing.assert_array_equal(y_true.get_shape(), [1, 3, 2])  # batch x output x (values, mask)
    y_pred = K.variable(np.array([[10, 20, 0.1], [104, 32, 145]]).T[None, :, :])
    loss = K.eval(_players.weighted_mean_squared_error(y_true, y_pred))
    np.testing.assert_equal(loss, [3.])


def test_make_final_activation():
    y = K.variable(np.array([[110, 120, -100]]))
    output = K.eval(_players.make_final_activation(y))
    np.testing.assert_equal(output, [[[10, 10], [20, 20], [-102, -102]]])


def test_make_basic_model_and_compile():
    model = _players.make_basic_model(input_shape=(35, 32))
    optimizer = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss=_players.weighted_mean_squared_error, optimizer=optimizer)


def test_basic_player():
    network = _players.BasicNetwork()
    players = [_players.NetworkPlayer(network) for _ in range(4)]
    _game.initialize_players_cards(players)
    board = _game.GameBoard([], [(0, 80, np.random.choice([l for l in "hdsc"]))])
    card = players[0].get_card_to_play(board)
    assert isinstance(card, _players._deck.Card), "Unexpected type {}".format(type(card))


@genty.genty
class PlayabilityTests(TestCase):

    @genty.genty_dataset(
        correct_playability=(1, 4),
        incorrect_playability=(0, 20.118),
    )
    def test_masked_mean_squared_error(self, acceptable, expected):
        y_pred = K.variable(np.array([[1, acceptable, 0], [10, 100, 1000]]).T[None, :, :])
        y_true = K.variable(np.array([[1, 1, 0], [12, -1, -1]]).T[None, :, :])
        np.testing.assert_array_equal(y_pred.get_shape(), [1, 3, 2])  # batch x output x (acceptabilities, values)
        loss = K.eval(_players.PlayabilityOutput.playability_error(y_true, y_pred))
        np.testing.assert_almost_equal(loss, [expected], decimal=3)

    def test_make_final_playability_layer(self):
        y = K.variable(np.array([[110, 120, -100]]))
        output = K.eval(_players.PlayabilityOutput.make_final_playability_layer(y))
        np.testing.assert_equal(output.shape, [1, 33, 2])
        assert np.min(output[:, :, 0]) >= 0
        assert np.max(output[:, :, 0]) <= 1

    def test_make_playability_reference_unplayed(self):
        value = 12
        output = _players.PlayabilityOutput.make_playability_reference(playable_cards=None, played_card=None, value=value)
        expected_playable = [0] * 32 + [1]
        np.testing.assert_array_equal(output[:, 0], expected_playable)
        expected_values = [-1] * 32 + [value]
        np.testing.assert_array_equal(output[:, 1], expected_values)
        # errors
        np.testing.assert_raises(TypeError, _players.prepare_learning_output, C("Qh"), None, value)
        np.testing.assert_raises(AssertionError, _players.prepare_learning_output, None, [C("Qh")], value)

    def test_make_playability_reference_played(self):
        value = 12
        output = _players.PlayabilityOutput.make_playability_reference(playable_cards=CL(["7h", "8h"]), played_card=C("7h"), value=value)
        expected_playable = [1, 1] + [0] * 31
        np.testing.assert_array_equal(output[:, 0], expected_playable)
        expected_values = [value] + [-1] * 32
        np.testing.assert_array_equal(output[:, 1], expected_values)
