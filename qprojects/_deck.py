#-*- coding: utf-8 -*
import numpy as np


class Suits:
    heart = "❤"
    diamonds = "♦"
    spades = "♠"
    clubs = "♣"


SUITS = (Suits.heart, Suits.diamonds, Suits.spades, Suits.clubs)
_SUIT_CONVERTER = dict(zip("hdsc", SUITS))
VALUES = ("7", "8", "9", "J", "Q", "K", "10", "A")
TRUMP_ORDER = ("7", "8", "Q", "K", "10", "A", "9", "J")
_POINTS = {"7": (0, 0),  # points provided as: (regular, trump)
           "8": (0, 0),
           "9": (0, 14),
           "J": (2, 20),
           "Q": (3, 3),
           "K": (4, 4),
           "10": (10, 10),
           "A": (11, 11)}


class Card:

    def __init__(self, value, suit):
        """Card of a deck
        
        Parameters
        ----------
        value: str
            The value of the card, among: "7", "8", "9", "J", "Q", "K", "10", "A"
        suit: str
            The suit of the card, among: "❤", "♦", "♠", "♣" (or for simplicity's sake: "h", "d", "s", "c")
        """
        suit = _SUIT_CONVERTER.get(suit, suit)  # to be able to use h d s or c as suit
        self._value_suit = value + suit
        assert suit in SUITS, 'Unknown suit "{}"'.format(suit)
        assert value in _POINTS, 'Unknown value "{}"'.format(value)

    @property
    def value(self):
        """Value of the card
        """
        return self._value_suit[:-1]

    @property
    def suit(self):
        """Suit of the card
        """
        return self._value_suit[-1]

    def get_points(self, trump_suit):
        """Number of points awarded by the card

        Parameters
        ----------
        trump_suit: str
            the trump suit selected for the game
        """
        return _POINTS[self._value_suit[:-1]][self._value_suit[-1] == trump_suit]

    def get_order(self, trump_suit):
        """Order of the card in the suit, considering the trump suit (trump suit order differs from others).

        Parameters
        ----------
        trump_suit: str
            the trump suit selected for the game
        """
        l = TRUMP_ORDER if self.suit == trump_suit else VALUES
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
    """A list of cards with convenient game related methods

    Parameters
    ----------
    cards: list
        a list of Cards
    trump_suit: str
        the selected trump suit (None if not yet selected)
    """

    def __init__(self, cards, trump_suit=None):
        super().__init__(cards)
        self.trump_suit = _SUIT_CONVERTER.get(trump_suit, trump_suit)
        if self.trump_suit is not None:
            assert self.trump_suit in SUITS, 'Unknown suit "{}"'.format(self.trump_suit)

    def count_points(self):
        """Counts the number of points in the list of cards
        """
        if self.trump_suit is None:
            raise RuntimeError("Counting points is now allowed if trump_suit is not provided")
        return sum(card.get_points(self.trump_suit) for card in self)

    def get_round_string(self):
        """Returns a string representing a round.
        This only work with 4-length CardList
        Brackets are used to show the highest card. Stars are used to highlight the trump cards.
        
        Example
        -------
        Round with heart trump:  '  Q♦     [ K❤ *]    9♦       J♠     '

        Note
        ----
        This function is for representation purpose. The spaces are set so as that all cards will be aligned
        if printed on several lines.
        """
        if len(self) != 4:
            raise RuntimeError("Expected 4 cards but only had {}.".format(len(self)))
        highest = self.get_highest_round_card()
        strings = []
        for card in self:
            h = card == highest
            params = ["[" if h else " ", "" if card.value == "10" else " ",
                      card, " *" if card.suit == self.trump_suit else "  ", "]" if h else " "]
            strings.append("{}{}{}{}{}  ".format(*params))
        return "".join(strings)

    def get_highest_round_card(self):
        """Returns the highest card considering the list is a round (first card provides the required suit)
        """
        if not self:
            return None
        if self.trump_suit is None:
            raise RuntimeError("Highest card cannot be identified with unspecified trump_suit")
        main_card = self[0]
        for card in self[1:]:
            if card.suit == self.trump_suit and main_card.suit != self.trump_suit:
                main_card = card
            elif main_card.suit == card.suit and main_card.get_order(self.trump_suit) < card.get_order(self.trump_suit):
                main_card = card
        return main_card

    def get_playable_cards(self, round_cards):
        """Returns all the cards from a hand which can be played in the round.

        Parameters
        ----------
        round_cards: list
            list of cards played in the round
        """
        if self.trump_suit is None:
            raise RuntimeError("Playable cards cannot be specified when trump_suit is not specified")
        if not round_cards:
            return self
        hand_cards_dict = {}
        for card in self:
            hand_cards_dict.setdefault(card.suit, []).append(card)
        # same suit case
        first_suit = round_cards[0].suit
        if first_suit in hand_cards_dict and first_suit != self.trump_suit:
            return hand_cards_dict[first_suit]
        # available trumps
        playable = hand_cards_dict.get(self.trump_suit, [])
        highest_card = round_cards.get_highest_round_card()
        if highest_card.suit == self.trump_suit:
            over = [c for c in playable if c.get_order(self.trump_suit) > highest_card.get_order(self.trump_suit)]
            playable = over if over else playable
        # other cards if no trump or partner lead: everything else is available
        if not playable or (first_suit != self.trump_suit and len(round_cards) - round_cards.index(highest_card) == 2):
            playable.extend([c for suit, cards in hand_cards_dict.items() for c in cards if suit != self.trump_suit])
        return CardList(playable, self.trump_suit)

