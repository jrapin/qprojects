#-*- coding: utf-8 -*
import numpy as np


class Suits:
    heart = "❤"
    diamonds = "♦"
    spades = "♠"
    clubs = "♣"


SUITS = (Suits.heart, Suits.diamonds, Suits.spades, Suits.clubs)
VALUES = ("7", "8", "9", "10", "J", "Q", "K", "A")
_POINTS = {"7": (0, 0),  # points provided as: (regular, trump)
           "8": (0, 0),
           "9": (0, 14),
           "10": (10, 10),
           "J": (2, 20),
           "Q": (3, 3),
           "K": (4, 4),
           "A": (11, 11)}


class Card:

    def __init__(self, value, suit):
        assert suit in SUITS
        assert value in _POINTS
        self._value_suit = value + suit

    def points(self, trump_suit):
        return _POINTS[self._value_suit[0]][self._value_suit[1] == trump_suit]

    def __hash__(self):
        return self._value_suit.__hash__()

    def __repr__(self):
        return self._value_suit

    def __eq__(self, other):
        if isinstance(other, str):
            return self._value_suit == other
        elif isinstance(other, self.__class__):
            return self._value_suit == other._value_suit
        else:
            raise NotImplementedError


class CardList(list):

    def __init__(self, cards):
        super().__init__(cards)

    def count_points(self, trump_suit):
        return sum(card.points(trump_suit) for card in self)


class DefaultPlayer:

    def __init__(self):
        self._cards = []

    @property
    def cards(self):
        return list(self._cards)

    @cards.setter
    def cards(self, cards):
        assert len(cards) == 8
        self._cards = cards


class Game:
    
    def __init__(self, players):
        self.players = players
        cards = [Card(v, s) for s in SUITS for v in VALUES]
        np.random.shuffle(cards)
        print(cards)
        for k in range(4):
            self.players[k].cards = CardList(cards[8 * k: 8 * (k + 1)])

