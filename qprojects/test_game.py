# -*- coding: utf-8 -*
from unittest import TestCase
from pathlib import Path
import tempfile
import genty
import numpy as np
from . import _game
from . import _deck
from ._deck import Card as C


def play_a_game(verbose=False):
    game = _game.Game([_game.DefaultPlayer() for _ in range(4)])
    game.board.biddings.append((0, 80, "❤"))
    game.trump_suit = "❤"
    game.play_game(verbose=verbose)
    return game


def test_game_initialization():
    # np.random.seed(12)
    game = _game.Game([_game.DefaultPlayer() for _ in range(4)])
    # check no duplicate
    playable = {c for p in game.players for c in p.cards}
    assert len(playable) == 32


def test_random_game():
    play_a_game(verbose=True)
    # raise Exception


def test_game_board_eq():
    board1 = _game.GameBoard([(1, C("9♦"))], [(1, 80, "♦")])
    board2 = _game.GameBoard([(1, C("9d"))], [(1, 80, "♦")])
    board3 = _game.GameBoard([(1, C("9h"))], [(1, 80, "♦")])
    board1.assert_equal(board2)
    np.testing.assert_raises(AssertionError, board1.assert_equal, board3)


def test_board_dump_and_load():
    cards = [C(v + s) for v in _deck.VALUES for s in _deck.SUITS]
    board1 = _game.GameBoard([(k % 4, c) for k, c in enumerate(cards)], [(1, 80, "♦")])
    with tempfile.TemporaryDirectory() as tmp:
        filepath = Path(tmp) / "board_dump_and_load_test.json"
        board1.dump(filepath)
        board2 = _game.GameBoard.load(str(filepath))
    board1.assert_equal(board2)


def test_game_board():
    filepath = Path(__file__).parent / "board_example.json"
    if not filepath.exists():  # TODO: commit it
        game = play_a_game()
        game.board.dump(filepath)
    board = _game.GameBoard.load(filepath)
    board.assert_valid()


@genty.genty
class GameTests(TestCase):

    played_cards = [(0, '10❤'), (1, '9❤'), (2, '7❤'), (3, 'J❤'), (3, 'K♦'), (0, 'Q♦'), (1, '8♦'), (2, '7♦'), (3, '8♣'),
                    (0, '7♣'), (1, '10♦'), (2, 'J♣'), (2, 'A♣'), (3, 'K♣'), (0, '10♣'), (1, '8❤'), (1, 'A♠'), (2, '9♠'),
                    (3, 'K♠'), (0, 'J♠'), (1, 'Q♠'), (2, '10♠'), (3, '7♠'), (0, 'K❤'), (0, 'A❤'), (1, '8♠'), (2, 'Q♣'),
                    (3, 'Q❤'), (0, '9♣'), (1, 'A♦'), (2, '9♦'), (3, 'J♦'), (0, 'None')]

    @genty.genty_dataset(
        correct=("❤", "", []),
        incorrect_trump=("♦", "Wrong player for round #1", []),
        unauthorized=("❤", 'Unauthorized Card("K♦") played by player 3', [(3, (3, 'K♦')), (4, (3, 'J❤'))]),
        duplicate=("❤", "Some cards are repeated", [(4, (3, 'J❤'))]),
    )
    def test_gameboard_assert_valid(self, trump_suit, expected, changes):
        played_cards = list(self.played_cards)  # duplicate the list
        for index, value in changes:
            played_cards[index] = value
        board = _game.GameBoard()
        board.played_cards = [(p, C(c)) for p, c in played_cards[:32]] + [(0, None)]
        board.biddings.append((0, 80, trump_suit))
        if not expected:
            board.assert_valid()
        else:
            try:
                board.assert_valid()
            except AssertionError as error:
                assert expected in error.args[0], 'Incorrect message "{}"\nExpected "{}"'.format(error.args[0], expected)
            else:
                raise AssertionError("An error should have been raised")

    @genty.genty_dataset(
        first_of_first=(0, []),
        first_of_second=(4, [0, 1, 2, 3]),
        first_of_finished=(-1, [28, 29, 30, 31]),
        first_of_finished_2=(32, [28, 29, 30, 31]),
        second_of_second=(5, [4]),
    )
    def test_get_last_round(self, index, expected_inds):
        board = _game.GameBoard()
        board.biddings.append((0, 80, "❤"))
        expected = [self.played_cards[ind][1] for ind in expected_inds]
        expected = _deck.CardList([C(c) for c in expected], "❤")
        played_cards = [(p, C(c)) for p, c in self.played_cards[:32]] + [(0, None)]
        board.played_cards = played_cards[:index]
        output = board.get_current_round_cards()
        output.assert_equal(expected)
