#-*- coding: utf-8 -*

SUITS = ('heart', 'diamonds', 'spades', 'clubs')
"❤", "♦"
VALUES = ("7", "8", "9", "10", "J", "Q", "K", "A")
_POINTS = {"7": (0, 0),  # points provided as: (regular, trump)
           "8", (0, 0),
           "9", (0, 14),
           "10", (10, 10),
           "J", (2, 20),
           "Q", (3, 3),
           "K", (4, 4),
           "A", (11, 11)}



class Card:

    def __init__(self, value, suit):
        self.value = value
        self.suit = suit

    def points(self, trump_color):
        return _POINTS[self.suit == trump_suit]

    def __hash__(self):
        return (self.value, self.suit).__hash__()


def CardList(list):

    def __init__(self, cards):
        super().__init__(cards)

    def count_points(self, trump_suit):
        return sum(card.points(trump_suit) for card in self)


class DefaultPlayer:

    def __init__(self)
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
        cards.shuffle()
        for k in range(4):
            self.players.cards = CardList(cards[8 * k: 8 * (k + 1)]




class CardRound(CardList):
    pass

