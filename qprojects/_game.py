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
        data = self._as_dict()
        filepath = Path(filepath)
        with filepath.open("w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath):
        instance = cls()
        filepath = Path(filepath)
        with filepath.open("r") as f:
            data = json.load(f)
        played_cards = [(p, _deck.Card(c[:-1], c[-1]) if c != 'None' else None) for p, c in data["played_cards"]]
        return cls(played_cards, [tuple(b) for b in data["biddings"]])

    @property
    def trump_suit(self):
        return self.biddings[-1][-1]

    def __repr__(self):
        return str(self._as_dict())

    def assert_equal(self, other):
        for name in ["biddings", "played_cards"]:
            for k, (element1, element2) in enumerate(zip(getattr(self, name), getattr(other, name))):
                if element1 != element2:
                    raise AssertionError("Discrepency with element #{} of {}: {} Vs {}".format(k, name, element1, element2))

    @property
    def is_complete(self):
        return len(self.played_cards) == 33  # 32 played and the additional 33rd element to record last winner

    def assert_valid(self):
        assert self.is_complete, "Game is not complete"
        assert len({x[1] for x in self.played_cards[:32]}) == 32, "Some cards are repeated"
        player_cards = [[] for _ in range(4)]
        for p_card in self.played_cards[:32]:
            player_cards[p_card[0]].append(p_card[1])            
        player_cards = [_deck.CardList(c, self.trump_suit) for c in player_cards]
        # check the sequence
        first_player = 0
        for k, round_cards in enumerate(_utils.grouper(self.played_cards[:32], 4)):
            # player order
            expected_players = (first_player + np.arange(4)) % 4
            players = [rc[0] for rc in round_cards]
            np.testing.assert_array_equal(players, expected_players, "Wrong players for round #{}".format(k))
            round_cards_list = _deck.CardList([x[1] for x in round_cards], self.trump_suit)
            first_player = (first_player + round_cards_list.index(round_cards_list.get_highest_round_card())) % 4            
            # cards played
            for k, player_card in enumerate(round_cards):
                visible_round = _deck.CardList(round_cards_list[:k], self.trump_suit)
                assert player_card[1] in player_cards[player_card[0]].get_playable_cards(visible_round), "Unauthorised card played"
                player_cards[player_card[0]].remove(player_card[1])
        assert first_player == self.played_cards[-1][0], "Wrong winner of last round"
        assert not any(x for x in player_cards), "Remaining cards, this function is improperly coded"



def extract_last_round(played_cards):
    start = (len(played_cards) // 4) * 4
    return Round([x[1] for x in played_cards[start:]])

