from pathlib import Path
import ujson as json
import numpy as np
from . import _deck


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

    def get_played_card(self, round_cards):
        playable = self._cards.get_playable_cards(round_cards)
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
        round_cards = _deck.CardList([], self.trump_suit)
        for k in range(4):
            player_ind = (first_player_index + k) % 4
            selected = self.players[player_ind].get_played_card(round_cards)
            round_cards.append(selected)
            self.board.played_cards.append((player_ind, selected))
        return round_cards

    def play_game(self, verbose=False):  # still buggy
        first_player_index = 0
        for k in range(1, 9):
            round_cards = self.play_round(first_player_index)
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

    def dump(self, filepath):
        data = self._as_dict()
        filepath = Path(filepath)
        with filepath.open("w") as f:
            json.dump(data, f)

    def _as_dict(self):
        data = {"played_cards": [(p, str(c)) for p, c in self.played_cards],
                "biddings": self.biddings}
        return data

    @classmethod
    def load(cls, filepath):
        instance = cls()
        filepath = Path(filepath)
        with filepath.open("r") as f:
            data = json.load(f)
        played_cards = [(p, _deck.Card(c[:-1], c[-1])) for p, c in data["played_cards"]]
        return cls(played_cards, [tuple(b) for b in data["biddings"]])

    def __repr__(self):
        return str(self._as_dict())

    def assert_equal(self, other):
        for name in ["biddings", "played_cards"]:
            for k, (element1, element2) in enumerate(zip(getattr(self, name), getattr(other, name))):
                if element1 != element2:
                    raise AssertionError("Discrepency with element #{} of {}: {} Vs {}".format(k, name, element1, element2))


def extract_last_round(played_cards):
    start = (len(played_cards) // 4) * 4
    return Round([x[1] for x in played_cards[start:]])

