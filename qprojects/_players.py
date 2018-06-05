import os
import numpy as np
import keras
from keras import backend as K
from . import _utils
from . import _deck
from . import _game


class NetworkPlayer(_game.DefaultPlayer):

    def __init__(self, network):
        super().__init__()
        self._network = network
        self._representation = None
        self._last_update_step = None

    def initialize_game(self, order, cards):
        super().initialize_game(order, cards)
        # representation: (1 initial cards + 1 order + 1 trump + 32 played cards) x 32 cards
        self._representation = np.zeros((35, 32))
        self._representation[0, :] = self._initial_cards.as_array()
        self._representation[1, order] = 1
        self._last_update_step = 0

    def _update_representation(self, board):
        self._representation[2, _deck.SUITS.index(board.trump_suit)] = 1
        for k, action in enumerate(board.actions[self._last_update_step:]):
            self._representation[3 + k + self._last_update_step, action[1].global_index] = 1
        self._last_update_step = len(board.actions)

    def _propose_card_to_play(self, board):
        self._update_representation(board)
        output = self._network.predict(self._representation)
        index = np.argmax(output[:32])
        return _deck.Card.from_global_index(index)

    def set_reward(self, board, value):
        previous_representation = np.array(self._representation, copy=True)
        self._update_representation(board)
        value = max(0, np.max(self._network.predict(self._representation)))
        if self._last_playable_cards is None:
            expected = prepare_learning_output(None, None, value)
        else:
            last_player, last_card = board.actions[-1]
            assert last_player == self._order
            expected = prepare_learning_output(self._last_playable_cards, last_card, value)
            self._last_playable_cards = None
        self._network.train(previous_representation, expected)


def prepare_learning_output(playable_cards, played_card, value, value_weight=5):
    """ 33x2 (32 cards + 1 unplayed) x (value and mask)
    """
    output = np.ones((33, 2))
    if playable_cards is None:
        assert played_card is None
        output[:, 0] = -100
        output[-1, 0] = value
        output[-1, 1] = value_weight
    else:
        playable_cards = _deck.CardList((c for c in playable_cards if c != played_card))
        output[:32, 1] -= playable_cards.as_array()
        output[:, 0] = -100
        output[played_card.global_index, 0] = value
        output[played_card.global_index, 1] = value_weight
    return output


def weighted_mean_squared_error(y_true, y_pred):
    """Mean squared error with weights

    Parameters
    ----------
    y_true: tensor
        [bach x outputs x 2] tensor of values (first of two index of last dimension) and masks (second)
    y_pred: tensor
        [batch x outputs] tensor of prediction
    """
    y_pred_values = y_pred[:, :, 0]
    y_true_values = y_true[:, :, 0]
    y_true_weights = y_true[:, :, 1]
    weighted_output = (y_pred_values - y_true_values) * y_true_weights
    return K.mean(K.square(weighted_output), axis=-1)


def make_final_activation(layer):
    output = keras.layers.LeakyReLU(alpha=0.02)(layer)
    output = keras.layers.Lambda(lambda x: x[:, :, None] - 100)(output)
    return keras.layers.concatenate([output, output], axis=2)


def make_basic_model(input_shape):
    input_data = keras.layers.Input(shape=input_shape)
    output = keras.layers.Flatten()(input_data)
    output = keras.layers.Dense(1200, activation=None, use_bias=False)(output)  # sparse input (avoid using bias)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.Dropout(.1)(output)
    output = keras.layers.Dense(1200, activation=None)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.Dropout(.1)(output)
    output = keras.layers.Dense(1024, activation=None)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.Dropout(.1)(output)
    output = keras.layers.Dense(512, activation=None)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.Dropout(.1)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.Dense(128, activation=None)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.Dropout(.1)(output)
    output = keras.layers.Dense(33, activation=None)(output)
    output = keras.layers.LeakyReLU(alpha=0.01)(output)
    output = keras.layers.Dense(33, activation=None)(output)
    output = make_final_activation(output)
    model = keras.models.Model(input_data, output)
    return model


@_utils.singleton
class BasicNetwork:

    def __init__(self, queue_size=1000, batch_size=16, verbose=0, model_filepath=None):
        self._batch_size = batch_size
        self._verbose = verbose
        self._queue = _utils.ReplayQueue(queue_size)
        if model_filepath is None or not os.path.exists(model_filepath):
            self._model = make_basic_model(input_shape=(35, 32))
        else:
            self._model = keras.models.load_model(model_filepath)
        optimizer = keras.optimizers.RMSprop(lr=0.00001, rho=0.9, epsilon=1e-08, decay=0.0)
        self._model.compile(loss=weighted_mean_squared_error, optimizer=optimizer)

    def predict(self, representation):
        return self._model.predict(representation[None, :, :])[0, :, 0]

    def train(self, representation, expected):
        batch_data = [(representation, expected)] + self._queue.get_random_selection(self._batch_size - 1)
        batch_representation, batch_expected = (np.array(x) for x in zip(*batch_data))
        self._model.fit(batch_representation, batch_expected, batch_size=self._batch_size, epochs=1, verbose=self._verbose)
        self._queue.append((representation, expected))

    def __del__(self):
        self._model.save("basic_network_last.h5")
