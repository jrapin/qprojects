# -*- coding: utf-8 -*
from unittest import TestCase
import genty
import numpy as np
from . import _deck
from . import _utils
from ._deck import Card as C


def test_card_equality_and_global_index():
    card1 = _deck.Card("K♦")
    card2 = _deck.Card("K♦")
    assert card1 == card2
    assert card1 == "K♦"
    assert card1 != None  # pylint: disable=singleton-comparison
    assert card1 != 3
    _ = _deck.Card("Q♦")
    card3 = _deck.Card(card2)
    assert card3 == card2
    np.testing.assert_equal(card1.global_index, 13)


def test_card_hash():
    card1 = _deck.Card("K♦")
    card2 = _deck.Card("K♦")
    card3 = _deck.Card("Q♦")
    assert {card1, card2, card3} == {card1, card3}


def test_card_points_suit_and_value():
    card = _deck.Card("J♦")
    np.testing.assert_equal(card.get_points("♦"), 20)
    np.testing.assert_equal(card.get_points("❤"), 2)
    np.testing.assert_equal(card.suit, "♦")
    np.testing.assert_equal(card.value, "J")
    card = _deck.Card("10d")
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


def test_full_deck_and_cardlist_as_array():
    deck = _deck.get_full_deck()
    np.testing.assert_equal(len(deck), 32)
    np.testing.assert_equal(len(set(deck)), 32)
    np.testing.assert_array_equal(deck.as_array(), [1 for _ in range(32)])


def test_global_index():
    np.random.seed()
    global_index = np.random.randint(32)
    card = _deck.Card.from_global_index(global_index)
    np.testing.assert_equal(card.global_index, global_index, err_msg="Card does not match global_index")


@genty.genty
class DeckTests(TestCase):

    @genty.genty_dataset(
        no_trump=("♣", "K♦"),
        first_trump=("♦", "9♦"),
        other_trump=("♠", "J♠"),
    )
    def test_get_highest_round_card(self, trump_suit, expected):
        cards = [C("9♦"), C("K♦"), C("Q♠"), C("J♠"), C("A❤")]
        cards = _deck.CardList(cards, trump_suit)
        highest_card = cards.get_highest_round_card()
        np.testing.assert_equal(highest_card, C(expected))

    @genty.genty_dataset(
        same_suit=(True, ["8♦", "Q♦"], ["9♦", "K♦"]),
        same_suit_trump_by_partner=(True, ["8❤", "Q♦"], ["A❤"]),
        same_suit_high_trump_by_partner=(True, ["J❤", "Q♦"], ["A❤", "7❤"]),
        no_card=(True, ["7♣", "8♣"], ["A❤", "7❤"]),
        no_card_with_lead=(True, ["9♣", "8♣"], ["A❤", "7❤", "9♦", "K♦", "Q♠", "J♠"]),
        no_card_with_trump=(True, ["8♣", "8❤"], ["A❤"]),
        no_card_with_high_trump=(True, ["8♣", "J❤"], ["A❤", "7❤"]),
        no_card_with_trump_lead=(True, ["8♣", "8❤", "9♣"], ["A❤", "9♦", "K♦", "Q♠", "J♠"]),
        no_card_no_trump=(False, ["8♣", "8❤"], ["9♦", "K♦", "Q♠", "J♠"]),
        first_no_trump=(False, [], ["9♦", "K♦", "Q♠", "J♠"]),
    )
    def test_get_playable_cards(self, has_trump, round_cards, expected):
        trump_suit = "❤"
        expected = _deck.CardList(expected, trump_suit=trump_suit)
        round_cards = _deck.CardList(round_cards, trump_suit)
        hand_cards = _deck.CardList(["9d", "Kd", "Qs", "Js"] + (["Ah", "7h"] if has_trump else []), trump_suit)
        playable = hand_cards.get_playable_cards(round_cards)
        _utils.assert_set_equal(playable, expected)
        # check that the order is deterministic
        playable.assert_equal(expected)

    @genty.genty_dataset(
        no_trump=(["Q♦", "K♠", "9♦", "J♠"], '[ Q♦  ]    K♠       9♦       J♠     '),
        with_trump=(["Q♦", "K❤", "9♦", "J♠"], '  Q♦     [ K❤ *]    9♦       J♠     '),
    )
    def test_get_round_string(self, cards, expected):
        trump_suit = "❤"
        cards = _deck.CardList(cards, trump_suit)
        np.testing.assert_equal(cards.get_round_string(), expected)

    @genty.genty_dataset(
        same=("♣", ["8♣", "A❤"], None),
        other_card=("♣", ["8♣", "Q❤"], AssertionError),
        other_trump=("❤", ["8♣", "A❤"], AssertionError),
        longer=("♣", ["8♣", "A❤", "Q❤"], AssertionError),
        shorter=("♣", ["8♣"], AssertionError),
    )
    def test_card_list_assert_equal(self, other_trump, other_cards, expected):
        round_cards = _deck.CardList(["8♣", "A❤"], "♣")
        other = _deck.CardList(other_cards, other_trump)
        if expected is None:
            round_cards.assert_equal(other)
        else:
            np.testing.assert_raises(expected, round_cards.assert_equal, other)
