#-*- coding: utf-8 -*
from unittest import TestCase
import genty
import numpy as np
from . import _deck
from . import _utils
from ._deck import Card as C


def test_card_equality():
    card1 = _deck.Card(*"K♦")
    card2 = _deck.Card(*"K♦")
    card3 = _deck.Card(*"Q♦")
    assert card1 == card2
    assert card1 == "K♦"
    card3 = _deck.Card(*"Q♦")
    np.testing.assert_raises(NotImplementedError, card1.__eq__, 3)


def test_card_hash():
    card1 = _deck.Card(*"K♦")
    card2 = _deck.Card(*"K♦")
    card3 = _deck.Card(*"Q♦")
    assert {card1, card2, card3} == {card1, card3}


def test_card_points_suit_and_value():
    card = _deck.Card(*"J♦")
    np.testing.assert_equal(card.get_points("♦"), 20)
    np.testing.assert_equal(card.get_points("❤"), 2)
    np.testing.assert_equal(card.suit, "♦")
    np.testing.assert_equal(card.value, "J")
    card = _deck.Card("10", "♦")
    np.testing.assert_equal(card.value, "10")
    np.testing.assert_equal(card.suit, "♦")
    np.testing.assert_equal(card.get_points("❤"), 10)


def test_cardlist_wrong_suit():
    np.testing.assert_raises(AssertionError, _deck.CardList, [], "x")


def test_get_highest_round_card_empty():
    cards = _deck.CardList([], "h")
    np.testing.assert_equal(cards.get_highest_round_card(), None)


def test_get_round_string_error():
    cards = _deck.CardList([], "h")
    np.testing.assert_raises(RuntimeError, cards.get_round_string)

    

@genty.genty
class DeckTests(TestCase):

    @genty.genty_dataset(
        no_trump=("♣", "K♦"),
        first_trump=("♦", "9♦"),
        other_trump=("♠", "J♠"),
    )
    def test_get_highest_round_card(self, trump_suit, expected):
        cards = [C(*"9♦"), C(*"K♦"), C(*"Q♠"), C(*"J♠"), C(*"A❤")]
        cards = _deck.CardList(cards, trump_suit)
        highest_card = cards.get_highest_round_card()
        np.testing.assert_equal(highest_card, C(*expected))


    @genty.genty_dataset(
        same_suit=(True, ["8♦", "Q♦"], ["9♦", "K♦"]),
        same_suit_trump_by_partner=(True, ["8❤", "Q♦"], ["A❤"]),
        same_suit_high_trump_by_partner=(True, ["J❤", "Q♦"], ["7❤", "A❤"]),
        no_card=(True, ["7♣", "8♣"], ["7❤", "A❤"]),
        no_card_with_lead=(True, ["9♣", "8♣"], ["7❤", "A❤", "K♦", "Q♠", "9♦", "J♠"]),
        no_card_with_trump=(True, ["8♣", "8❤"], ["A❤"]),
        no_card_with_high_trump=(True, ["8♣", "J❤"], ["7❤", "A❤"]),
        no_card_with_trump_lead=(True, ["8♣", "8❤", "9♣"], ["A❤", "K♦", "Q♠", "9♦", "J♠"]),
        no_card_no_trump=(False, ["8♣", "8❤"], ["K♦", "Q♠", "9♦", "J♠"]),
        first_no_trump=(False, [], ["K♦", "Q♠", "9♦", "J♠"]),
    )
    def test_get_playable_cards(self, has_trump, round_cards, expected):
        trump_suit =  "❤"
        expected = [C(*x) for x in expected]
        round_cards = _deck.CardList([C(*x) for x in round_cards], trump_suit)
        hand_cards = [C(*"9♦"), C(*"K♦"), C(*"Q♠"), C(*"J♠")] + ([C(*"A❤"), C(*"7❤")] if has_trump else [])
        hand_cards = _deck.CardList(hand_cards, trump_suit)
        playable = hand_cards.get_playable_cards(round_cards)
        _utils.assert_set_equal(playable, expected)

    @genty.genty_dataset(
        no_trump=(["Q♦", "K♠", "9♦", "J♠"], '[ Q♦  ]    K♠       9♦       J♠     '),
        with_trump=(["Q♦", "K❤", "9♦", "J♠"], '  Q♦     [ K❤ *]    9♦       J♠     '),
    )
    def test_get_round_string(self, cards, expected):
        trump_suit =  "❤"
        cards = _deck.CardList([C(s[:-1], s[-1]) for s in cards], trump_suit)
        np.testing.assert_equal(cards.get_round_string(), expected)

