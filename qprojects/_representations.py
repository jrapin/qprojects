import numpy as np
from . import _deck


class ExplicitRepresentation:

    shape = (34, 36)
    # representation: (1 initial cards and order + 1 trump + 32 played cards) x (32 cards + 4 order)

    @classmethod
    def create(cls, board, player, no_last=False):
        # player specific
        representation = np.zeros(cls.shape)
        representation[0, :32] = player.initial_cards.as_array()
        representation[0, 32 + player.order] = 1
        suit_ind = 8 * _deck.SUITS.index(board.trump_suit)
        representation[1, suit_ind: suit_ind + 8] = 1
        for k, (p_index, card) in enumerate(board.actions[:-1 if no_last else None]):
            representation[2 + k, card.global_index] = 1
            representation[2 + k, 32 + p_index] = 1
        return representation


class RoundRepresentation:  # current problem: "no_last" is complicated to implement

    shape = (7, 36)
    # representation: (1 current cards + 1 trump + 1 all played cards + 4 round cards with player) x (32 cards + 4 order)

    @classmethod
    def create(cls, board, player):
        # player specific
        representation = np.zeros(cls.shape)
        representation[0, :32] = player.cards.as_array()
        # trump
        suit_ind = 8 * _deck.SUITS.index(board.trump_suit)
        representation[1, suit_ind: suit_ind + 8] = 1
        # played cards
        played = _deck.CardList([a[1] for a in board.actions])
        representation[2, :32] = played.as_array()
        # round
        round_cards = board.get_current_round_cards()
        round_cards = [] if len(round_cards) == 4 else round_cards
        player_ind = board.next_player if not round_cards else round_cards[0][0]
        for k in range(4):
            representation[3 + k, 32 + player_ind] = 1
            if k < len(round_cards):
                representation[3 + k, round_cards[k].global_index] = 1
            player_ind += 1
        return representation
