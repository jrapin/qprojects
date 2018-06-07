import numpy as np
from . import _deck


class ExplicitRepresentation:

    shape = (34, 36)
    # representation: (1 initial cards and order + 1 trump + 32 played cards) x 32 cards

    @classmethod
    def create(cls, board, player, no_last=False):
        # player specific
        representation = np.zeros(cls.shape)
        representation[0, :32] = player.initial_cards.as_array()
        representation[0, 32 + player.order] = 1
        representation[1, _deck.SUITS.index(board.trump_suit)] = 1
        for k, (p_index, card) in enumerate(board.actions[:-1 if no_last else None]):
            representation[2 + k, card.global_index] = 1
            representation[2 + k, 32 + p_index] = 1
        return representation
