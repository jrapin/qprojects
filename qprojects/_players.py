import numpy as np
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
        self._representation[2, _deck.SUITS.order(board.trump)] = 1
        for k, action in board.actions[self._last_update_step:]:
            self._representation[3 + k + self._last_update_step, action[1].global_index] = 1
        self._last_update_step = len(board.actions)

    def _propose_card_to_play(self, board):
        self._update_representation(board)
        output = self._nework.predict(self._representation)
        index = np.argmax(output[:32])
        return _deck.Card.from_global_index(index)

    def set_reward(self, board, value):
        self._update_representation(board)
        expected = -100 * np.ones((33,))
        if self._last_playable_cards is None:
            expected[-1] = max(0, max(self._network.predict(self._representation)))
        else:
            action = board.actions[-1]
            assert action[0] == self.order
            mask = 1 - self._last_playable_cards.as_array()
            mask[action[1].global_index] = 1
        self._last_playable_cards = None


def prepare_learning_output(playable_cards, played_card, value):
    """ 33x2 (32 cards + 1 unplayed) x (value and mask)
    """
    output = np.ones((33, 2))
    if playable_cards is None:
        assert played_card is None
        output[:, 0] = -100
        output[-1, 0] = value
    else:
        playable_cards = _deck.CardList((c for c in playable_cards if c != played_card))
        output[:32, 1] -= playable_cards.as_array()
        output[:, 0] = -100
        output[played_card.global_index, 0] = value
    return output
