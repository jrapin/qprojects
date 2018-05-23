#-*- coding: utf-8 -*
import tempfile
from pathlib import Path
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


def test_game_initialization():
    #np.random.seed(12)
    game = _deck.Game([_deck.DefaultPlayer() for _ in range(4)])
    # check no duplicate
    playable = {c for p in game.players for c in p.cards}
    assert len(playable) == 32
    

@genty.genty
class GameTests(TestCase):

    @genty.genty_dataset(
        no_trump=("♣", "K♦"),
        first_trump=("♦", "9♦"),
        other_trump=("♠", "J♠"),
    )
    def test_get_highest_card(self, trump_suit, expected):
        cards = [C(*"9♦"), C(*"K♦"), C(*"Q♠"), C(*"J♠"), C(*"A❤")]
        highest_card = _deck.get_highest_card(cards, trump_suit)
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
        fist_no_trump=(False, [], ["K♦", "Q♠", "9♦", "J♠"]),
    )
    def test_get_playable_cards(self, has_trump, round_cards, expected):
        expected = [C(*x) for x in expected]
        round_cards = [C(*x) for x in round_cards]
        hand_cards = [C(*"9♦"), C(*"K♦"), C(*"Q♠"), C(*"J♠")] + ([C(*"A❤"), C(*"7❤")] if has_trump else [])
        playable = _deck.get_playable_cards(hand_cards, round_cards, "❤")
        _utils.assert_set_equal(playable, expected)


def test_game():
    game = _deck.Game([_deck.DefaultPlayer() for _ in range(4)])
    game.trump_suit = "❤"
    game.play_game(verbose=True)
    raise Exception


def test_game_board_eq():
        board1 = _deck.GameBoard([(1, C(*"9♦"))], [(1, 80, "♦")])
        board2 = _deck.GameBoard([(1, C(*"9d"))], [(1, 80, "♦")])
        board3 = _deck.GameBoard([(1, C(*"9h"))], [(1, 80, "♦")])
        board1.assert_equal(board2)
        np.testing.assert_raises(AssertionError, board1.assert_equal, board3)


def test_board_dump_and_load():
    cards = [_deck.Card(v, s) for v in _deck.VALUES for s in _deck.SUITS]
    board1 = _deck.GameBoard([(k % 4, c) for k, c in enumerate(cards)], [(1, 80, "♦")])
    with tempfile.TemporaryDirectory() as tmp:
        filepath = Path(tmp) / "board_dump_and_load_test.json"
        board1.dump(filepath)
        board2 = _deck.GameBoard.load(str(filepath))
    board1.assert_equal(board2)

