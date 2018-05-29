#-*- coding: utf-8 -*
from pathlib import Path
import itertools
import operator
import ujson as json
import numpy as np
from . import _deck
from . import _utils


class DefaultPlayer:
    """Player which selects one card randomly at each round
    """

    def __init__(self):
        self._cards = []

    @property
    def cards(self):
        """Makes a copy, to make sure no modification happens outside
        """
        return _deck.CardList(self._cards, self._cards.trump_suit)

    @cards.setter
    def cards(self, cards):
        assert len(cards) == 8
        self._cards = cards

    def get_card_to_play(self, board):
        round_cards = board.get_current_round_cards()
        playable = self._cards.get_playable_cards([] if len(round_cards) ==  4 else round_cards)
        selected = np.random.choice(playable)
        self._cards.remove(selected)
        return selected


class Game:
    
    def __init__(self, players):
        self.players = players
        self._trump_suit = None
        self.board = GameBoard()
        self.points = np.zeros((2, 9))
        self.initialize()

    def initialize(self):
        cards = [_deck.Card(v, s) for s in _deck.SUITS for v in _deck.VALUES]
        np.random.shuffle(cards)
        for k in range(4):
            self.players[k].cards = _deck.CardList(cards[8 * k: 8 * (k + 1)])
            if len({_deck.Card(*"K❤"), _deck.Card(*"Q❤")} & set(self.players[k].cards)) == 2:
                self.points[k % 2, -1] = 20

    @property
    def trump_suit(self):
        return self._trump_suit

    @trump_suit.setter
    def trump_suit(self, trump_suit):
        self._trump_suit = trump_suit
        for player in self.players:
            player._cards.trump_suit = trump_suit  # bypass "cards" protection

    def play_round(self, first_player_index):
        if self.trump_suit is None:
            raise RuntimeError("Trump suit should be specified")
        for k in range(4):
            player_ind = (first_player_index + k) % 4
            selected = self.players[player_ind].get_card_to_play(self.board)
            self.board.played_cards.append((player_ind, selected))

    def play_game(self, verbose=False):
        first_player_index = 0
        for k in range(1, 9):
            self.play_round(first_player_index)
            round_cards = self.board.get_current_round_cards()
            assert len(round_cards) == 4
            highest_card = round_cards.get_highest_round_card()
            winner = round_cards.index(highest_card)
            points = round_cards.count_points() + (10 if k == 8 else 0)
            next_player_index = (winner + first_player_index) % 4
            self.points[next_player_index % 2, k - 1] = points
            if verbose:
                print("Round #{} - Player {} starts: {}  ({} points)".format(k, first_player_index, round_cards.get_round_string(), points))
            first_player_index = next_player_index
        if verbose:
            print(self.points)
        self.board.played_cards.append((first_player_index, None))


class GameBoard:
    """Elements which are visible to all players.

    Attributes
    ----------
    played_cards: list
        played cards, as a list of tuples of type (#player, card)
    biddings: list
        the sequence of biddings, as a list of tuples of type (#player, points, trump_suit)
    """

    def __init__(self, played_cards=None, biddings=None):
        self.played_cards = [] if played_cards is None else played_cards
        self.biddings = [] if biddings is None else biddings

    def _as_dict(self, with_letter=False):
        data = {"played_cards": [(p, str(c)) for p, c in self.played_cards],
                "biddings": self.biddings}
        return data

    def dump(self, filepath):
        """Dumps a GameBoard to a file

        Parameter
        ---------
        filepath: str or Path
            path to the file where to save the GameBoard.
        """
        data = self._as_dict()
        filepath = Path(filepath)
        with filepath.open("w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath):
        """Loads a GameBoard from a file

        Parameter
        ---------
        filepath: str or Path
            path to the file where the GameBoard is save.

        Returns
        -------
        GameBoard
            the loaded GameBoard
        """
        instance = cls()
        filepath = Path(filepath)
        with filepath.open("r") as f:
            data = json.load(f)
        played_cards = [(p, _deck.Card(c[:-1], c[-1]) if c != 'None' else None) for p, c in data["played_cards"]]
        return cls(played_cards, [tuple(b) for b in data["biddings"]])

    @property
    def trump_suit(self):
        """Selected trump suit for the game
        """
        return self.biddings[-1][-1]

    def __repr__(self):
        return str(self._as_dict())

    def assert_equal(self, other):
        """Asserts that the board is identical to the provided other board.
        """
        for name in ["biddings", "played_cards"]:
            for k, (element1, element2) in enumerate(zip(getattr(self, name), getattr(other, name))):
                if element1 != element2:
                    raise AssertionError("Discrepency with element #{} of {}: {} Vs {}".format(k, name, element1, element2))

    @property
    def is_complete(self):
        """Returns whether the game is complete
        The game is considered complete when all 32 cards are played and the 33rd element provides
        the winner of the last round.
        """
        return len(self.played_cards) == 33  # 32 played and the additional 33rd element to record last winner

    def assert_valid(self):
        """Asserts that the whole sequence is complete and corresponds to a valid game.
        """
        assert self.is_complete, "Game is not complete"
        assert len({x[1] for x in self.played_cards[:32]}) == 32, "Some cards are repeated"
        cards_by_player = [[] for _ in range(4)]
        for p_card in self.played_cards[:32]:
            cards_by_player[p_card[0]].append(p_card[1])            
        cards_by_player = [_deck.CardList(c, self.trump_suit) for c in cards_by_player]
        # check the sequence
        first_player = 0
        for k, round_played_cards in enumerate(_utils.grouper(self.played_cards[:32], 4)):
            # player order
            expected_players = (first_player + np.arange(4)) % 4
            players = [rc[0] for rc in round_played_cards]
            np.testing.assert_array_equal(players, expected_players, "Wrong player for round #{}".format(k))
            round_cards_list = _deck.CardList([x[1] for x in round_played_cards], self.trump_suit)
            first_player = (first_player + round_cards_list.index(round_cards_list.get_highest_round_card())) % 4            
            # cards played
            for k, (player, card) in enumerate(round_played_cards):
                visible_round = _deck.CardList(round_cards_list[:k], self.trump_suit)
                error_msg = "Unauthorized card {} played by player {}".format(card, player)
                assert card in cards_by_player[player].get_playable_cards(visible_round), error_msg
                cards_by_player[player].remove(card)
        # last winner and function check
        assert first_player == self.played_cards[-1][0], "Wrong winner of last round"
        assert not any(x for x in cards_by_player), "Remaining cards, this function is improperly coded"

    def get_current_round_cards(self):
        """Return the cards for the current round (or the round just played if all 4 cards have been played)
        """
        end = min(len(self.played_cards), 32)
        start = max(0, ((end - 1) // 4)) * 4
        return _deck.CardList([x[1] for x in self.played_cards[start: end]], self.trump_suit)

