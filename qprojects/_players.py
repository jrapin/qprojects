import os
import numpy as np
import keras
from keras import backend as K
from . import _utils
from . import _deck
from . import _game
from . import _representations


class NetworkPlayer(_game.DefaultPlayer):

    def __init__(self, network):
        super().__init__()
        self._network = network
        self._last_representation = None

    def _get_expectations(self, board):
        representation = self._network.representation.create(board, self)
        output = self._network.predict(representation)
        return PenaltyOutput.prepare_expectations(output)

    def _propose_card_to_play(self, board):
        output = self._get_expectations(board)
        index = np.argmax(output[:32])
        return _deck.Card.from_global_index(index)

    def set_reward(self, board, value):
        last_player, last_card = board.actions[-1]
        if np.random.rand() < .3:
            value = min(250, value + max(np.max(self._get_expectations(board)), 0))
            previous_representation = self._network.representation.create(board, self, no_last=True)
            if last_player == self.order:
                expected = PenaltyOutput.make_reference(self._last_playable_cards, last_card, value)
                self._network.train(previous_representation, expected)
            elif np.random.rand() < .25:
                expected = PenaltyOutput.make_reference(None, None, value)
                self._network.train(previous_representation, expected)


class PenaltyOutput:

    @staticmethod
    def make_reference(playable_cards, played_card, value, value_weight=6):
        """ 33x2 (32 cards + 1 unplayed) x (value and mask)
        """
        output = np.ones((33, 2))
        if playable_cards is None:
            assert played_card is None
            output[:, 0] = -100
            output[-1, 0] = value
            output[-1, 1] = value_weight / 4.
        else:
            playable_cards = _deck.CardList((c for c in playable_cards if c != played_card))
            output[:32, 1] -= playable_cards.as_array()
            output[:, 0] = -100
            output[played_card.global_index, 0] = value
            output[played_card.global_index, 1] = value_weight
        return output

    @staticmethod
    def error(y_true, y_pred):
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

    @staticmethod
    def _make_final_activation(layer):
        output = keras.layers.LeakyReLU(alpha=0.02)(layer)
        output = keras.layers.Lambda(lambda x: x[:, :, None] - 100)(output)
        return keras.layers.concatenate([output, output], axis=2)

    @staticmethod
    def make_final_layer(layer):
        output = keras.layers.Dense(33, activation=None)(layer)
        return PenaltyOutput._make_final_activation(output)

    @staticmethod
    def prepare_expectations(data):
        return data[:, 0]


class PlayabilityOutput:

    @staticmethod
    def make_final_layer(layer):
        """new layer predicting acceptatbilities and values
        """
        playabilities = keras.layers.Dense(33, activation="sigmoid")(layer)
        values = keras.layers.Dense(33, activation="relu")(layer)
        values = keras.layers.LeakyReLU(alpha=0.01)(values)
        playabilities = keras.layers.Lambda(lambda x: x[:, :, None])(playabilities)
        values = keras.layers.Lambda(lambda x: x[:, :, None])(values)
        values = keras.layers.Multiply()([playabilities, values])
        return keras.layers.concatenate([playabilities, values], axis=2)

    @staticmethod
    def error(y_true, y_pred):
        """Error mixing a crossentropy for playability classification and squared error for the expectation value

        Parameters
        ----------
        y_true: tensor
            [bach x num cards x 2] tensor of playable cards and expectation value for the played card (-1 otherwise)
        y_pred: tensor
            [batch x num cards x 2] tensor of predicted playable cards and expectation values
        """
        y_pred_playabilities = y_pred[:, :, 0]
        y_true_playabilities = y_true[:, :, 0]
        y_pred_values = y_pred[:, :, 1]
        y_true_values = y_true[:, :, 1]
        mask = K.cast(K.greater(y_true_values, -1), 'float32')
        masked_values_error = (y_pred_values - y_true_values) * mask
        values_error = K.sum(K.square(masked_values_error), axis=-1)
        acceptabilities_error = 100 * K.sum(K.binary_crossentropy(y_true_playabilities, y_pred_playabilities), axis=-1)
        return values_error + acceptabilities_error

    @staticmethod
    def make_reference(playable_cards, played_card, value):
        """ 33x2 (32 cards + 1 unplayed) x (value and mask)
        """
        output = np.zeros((33, 2))
        if playable_cards is None:
            assert played_card is None
            output[-1, 0] = 1
            output[-1, 1] = value
        else:
            output[:32, 0] = playable_cards.as_array()
            output[output[:, 0] > 0, 1] = -1
            index = played_card.global_index
            output[index, 1] = value
            assert output[index, 0] == 1
        return output

    @staticmethod
    def prepare_expectations(net_output):
        """Postprocess the network output so as to recover expectations,
        with -1 for rejected cards, and max(0, value) for accepted cards.
        """
        rejected = net_output[:, 0] < 0.5
        accepted = np.logical_not(rejected)
        expectations = np.array(net_output[:, 1])
        expectations[rejected] = -1
        expectations[accepted] = np.maximum(expectations[accepted], 0)
        return expectations


def make_basic_network(input_layer):
    output = keras.layers.Flatten()(input_layer)
    output = keras.layers.Dense(1300, activation=None, use_bias=False)(output)  # sparse input (avoid using bias)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.Dropout(.1)(output)
    output = keras.layers.Dense(1300, activation=None)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.Dropout(.1)(output)
    output = keras.layers.Dense(1024, activation=None)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.Dropout(.1)(output)
    output = keras.layers.Dense(512, activation=None)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    output = keras.layers.Dropout(.05)(output)
    output = keras.layers.Dense(128, activation=None)(output)
    output = keras.layers.LeakyReLU(alpha=0.3)(output)
    return output


class BasicNetwork:

    representation = _representations.ExplicitRepresentation
    output_framework = PenaltyOutput

    def __init__(self, queue_size=1000, batch_size=16, verbose=0, model_filepath=None, learning_rate=0.00001):  # pylint: disable=too-many-arguments
        self._batch_size = batch_size
        self._verbose = verbose
        self._queue = _utils.ReplayQueue(queue_size)
        if model_filepath is None or not os.path.exists(model_filepath):
            input_data = keras.layers.Input(shape=self.representation.shape)
            temp = make_basic_network(input_data)
            output = self.output_framework.make_final_layer(temp)
            self.model = keras.models.Model(input_data, output)
        else:
            self.model = keras.models.load_model(model_filepath, custom_objects={"error", self.output_framework.error})
        optimizer = keras.optimizers.SGD(lr=learning_rate)
        self.model.compile(loss=self.output_framework.error, optimizer=optimizer)

    def predict(self, representation):
        output = self.model.predict(representation[None, :, :])[0, :, :]
        if np.any(np.isnan(output)):
            raise RuntimeError("Nan values")
        return output

    def train(self, representation, expected):
        self._queue.append((representation, expected))
        batch_data = self._queue.get_random_selection(self._batch_size)
        batch_representation, batch_expected = (np.array(x) for x in zip(*batch_data))
        self.model.fit(batch_representation, batch_expected, batch_size=self._batch_size, epochs=1, verbose=self._verbose)
