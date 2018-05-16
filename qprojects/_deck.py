#-*- coding: utf-8 -*
import numpy as np


class Suits:
    heart = "❤"
    diamonds = "♦"
    spades = "♠"
    clubs = "♣"


SUITS = (Suits.heart, Suits.diamonds, Suits.spades, Suits.clubs)
VALUES = ("7", "8", "9", "10", "J", "Q", "K", "A")
TRUMP_ORDER = ("7", "8", "Q", "K", "10", "A", "9", "J")
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
        assert suit in SUITS, 'Unknown suit "{}"'.format(suit)
        assert value in _POINTS, 'Unknown value "{}"'.format(value)
        self._value_suit = value + suit

    @property
    def value(self):
        return self._value_suit[0]

    @property
    def suit(self):
        return self._value_suit[1]

    def get_points(self, trump_suit):
        return _POINTS[self._value_suit[0]][self._value_suit[1] == trump_suit]

    def get_order(self, trump_suit):
        return (TRUMP_ORDER if self.suit == trump_suit else VALUES).index(self.value)  # to be tested
        

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
        return sum(card.get_points(trump_suit) for card in self)


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


def get_highest_card(cards, trump_suit):
    """Returns the highest card among a set of cards

    Parameters
    ----------
    cards: list
        list of cards
    trump_suit: str
        trump suit
    """
    main_card = cards[0]
    for card in cards[1:]:
        if card.suit == trump_suit and main_card.suit != trump_suit:
            main_card = card
        elif main_card.suit == card.suit and main_card.get_order(trump_suit) < card.get_order(trump_suit):
            main_card = card
    return main_card
    
    
def get_playable_cards(hand_cards, round_cards, trump_suit):
    """Returns all the cards from a hand which can be played in the round.

    Parameters
    ----------
    hand_cards: list
        list of cards in the hand
    round_cards: list
        list of cards played in the round
    trump_suit: str
        trump suit
    """
    if not round_cards:
        return hand_cards
    hand_cards_dict = {}
    for card in hand_cards:
        hand_cards_dict.setdefault(card.suit, []).append(card)
    # same suit case
    first_suit = round_cards[0].suit
    if first_suit in hand_cards_dict and first_suit != trump_suit:
        return hand_cards_dict[first_suit]
    # available trumps
    playable = hand_cards_dict.get(trump_suit, [])
    highest_card = get_highest_card(round_cards, trump_suit)
    if highest_card.suit == trump_suit:
        playable = [c for c in playable if c.get_order(trump_suit) > highest_card.get_order(trump_suit)]
    # other cards if no trump or partner lead: everything else is available
    if not playable or (first_suit != trump_suit and len(round_cards) - round_cards.index(highest_card) == 2):
        playable.extend([c for suit, cards in hand_cards_dict.items() for c in cards if suit != trump_suit])
    return playable

