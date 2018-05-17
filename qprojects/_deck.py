#-*- coding: utf-8 -*
import numpy as np


class Suits:
    heart = "❤"
    diamonds = "♦"
    spades = "♠"
    clubs = "♣"


SUITS = (Suits.heart, Suits.diamonds, Suits.spades, Suits.clubs)
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
        self._value_suit = value + suit
        assert suit in SUITS, 'Unknown suit "{}"'.format(suit)
        assert value in _POINTS, 'Unknown value "{}"'.format(value)

    @property
    def value(self):
        return self._value_suit[:-1]

    @property
    def suit(self):
        return self._value_suit[-1]

    def get_points(self, trump_suit):
        return _POINTS[self._value_suit[0]][self._value_suit[1] == trump_suit]

    def get_order(self, trump_suit):
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

    def get_played_card(self, round_cards, trump_suit):
        playable = get_playable_cards(self._cards, round_cards, trump_suit)
        selected = np.random.choice(playable)
        self._cards.remove(selected)
        return selected


class Round(CardList):

    def __init__(self, starting_player, trump_suit):
        super().__init__([])
        self.starting_player = starting_player
        self.trump_suit = trump_suit

    def count_points(self):
        return super().count_points(self.trump_suit)

    def __str__(self):
        highest = get_highest_card(self, self.trump_suit)
        strings = ["Player #{} starts:   ".format(1 + self.starting_player)]
        for card in self:
            h = card == highest
            params = ["[" if h else " ", "" if card.value == "10" else " ",
                      card, " *" if card.suit == self.trump_suit else "  ", "]" if h else " "]
            strings.append("{}{}{}{}{}  ".format(*params))
        return "".join(strings)



class Game:
    
    def __init__(self, players):
        self.players = players
        self.trump_suit = None
        self.biddings = []
        self.rounds = []
        self.initialize()

    def initialize(self):
        cards = [Card(v, s) for s in SUITS for v in VALUES]
        np.random.shuffle(cards)
        for k in range(4):
            self.players[k].cards = CardList(cards[8 * k: 8 * (k + 1)])

    def play_round(self, first_player_index):
        if self.trump_suit is None:
            raise RuntimeError("Trump suit should be specified")
        round_cards = Round(first_player_index, self.trump_suit)
        for k in range(4):
            index = (first_player_index + k) % 4
            selected = self.players[index].get_played_card(round_cards, self.trump_suit)
            round_cards.append(selected)
        return round_cards

    def play_game(self, verbose=False):  # still buggy
        first_player_index = 0
        for k in range(1, 9):
            round_cards = self.play_round(first_player_index)
            highest_card = get_highest_card(round_cards, self.trump_suit)
            winner = round_cards.index(highest_card)
            if verbose:
                print("Round #{} - {}".format(k, str(round_cards)))
            first_player_index = (winner + first_player_index) % 4
                

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
        over = [c for c in playable if c.get_order(trump_suit) > highest_card.get_order(trump_suit)]
        playable = over if over else playable
    # other cards if no trump or partner lead: everything else is available
    if not playable or (first_suit != trump_suit and len(round_cards) - round_cards.index(highest_card) == 2):
        playable.extend([c for suit, cards in hand_cards_dict.items() for c in cards if suit != trump_suit])
    return playable

