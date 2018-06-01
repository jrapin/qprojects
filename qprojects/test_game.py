# -*- coding: utf-8 -*
from unittest import TestCase
from pathlib import Path
import tempfile
import genty
import numpy as np
from . import _game
from . import _deck
from . import _utils
from ._deck import Card as C


_PLAYED_CARDS = ((0, '10❤'), (1, '9❤'), (2, '7❤'), (3, 'J❤'), (3, 'K♦'), (0, 'Q♦'), (1, '8♦'), (2, '7♦'), (3, '8♣'),
                 (0, '7♣'), (1, '10♦'), (2, 'J♣'), (2, 'A♣'), (3, 'K♣'), (0, '10♣'), (1, '8❤'), (1, 'A♠'), (2, '9♠'),
                 (3, 'K♠'), (0, 'J♠'), (1, 'Q♠'), (2, '10♠'), (3, '7♠'), (0, 'K❤'), (0, 'A❤'), (1, '8♠'), (2, 'Q♣'),
                 (3, 'Q❤'), (0, '9♣'), (1, 'A♦'), (2, '9♦'), (3, 'J♦'))


def play_a_game(verbose=False):
    players = [_game.DefaultPlayer() for _ in range(4)]
    _game.initialize_players_cards(players)
    board = _game.GameBoard([], [(0, 80, "❤")])
    _game.play_game(board, players, verbose=verbose)
    return board


def test_game_initialization():
    players = [_game.DefaultPlayer() for _ in range(4)]
    _game.initialize_players_cards(players)
    # check no duplicate
    playable = {c for p in players for c in p.cards}
    assert len(playable) == 32
    np.testing.assert_array_equal([p._order for p in players], [0, 1, 2, 3],
                                  err_msg="Order is not correctly initialized")


def test_game_points_global():
    np.random.seed(12)
    players = [_game.DefaultPlayer() for _ in range(4)]
    special = ['Q❤', 'K❤']
    cards = special + [x[1] for x in _PLAYED_CARDS if x[1] not in special]
    cards = [C(x) for x in cards]
    for k, cards in enumerate(_utils.grouper(cards, 8)):
        players[k].initialize_game(k, _deck.CardList(cards))
    board = _game.GameBoard([], [(0, 80, '❤')])
    _game.play_game(board, players, verbose=True)
    expected = np.zeros((2, 32), dtype=int)
    expected[:, 3::4] = np.array([[25, 16, 5, 8, 0, 16, 40, 35],
                                  [0, 0, 0, 0, 17, 0, 0, 0]], dtype=int)
    np.testing.assert_array_equal(board.points[:, 3::4], expected[:, 3::4])
    np.testing.assert_equal(board.points.sum(), 182)
    expected[0, 21] = 20  # bonus
    np.testing.assert_array_equal(board.points, expected, "Missing bonus points")
    # replay with different trump
    mixer = {"❤": "♦", "♦": "❤"}
    actions = [(p, c.value + mixer.get(c.suit, c.suit)) for p, c in board.actions]
    board2 = _game.GameBoard(actions, [(0, 80, "♦")])
    board2.assert_valid()
    np.testing.assert_equal(board2.points.sum(), 162, "There should not be any bonus this time")
    expected[0, 21] = 0  # bonus
    np.testing.assert_array_equal(board2.points, expected, "Are there bonus points?")


def test_game_points_local_and_default_player():
    players = [_game.DefaultPlayer() for _ in range(4)]
    board = _game.GameBoard(_PLAYED_CARDS, [(0, 80, "h")])
    for k, cards in enumerate(board.replay_cards_iterator()):
        players[k].initialize_game(k, cards)
    board = _game.GameBoard([], [(0, 80, '❤')])
    _game.play_game(board, players, verbose=True)
    np.testing.assert_equal(players[0].reward_sum, 74)
    np.testing.assert_equal(players[1].reward_sum, 88)
    rewards = [players[2].reward_sum, players[3].reward_sum]
    np.testing.assert_array_equal(rewards, board.points.sum(axis=1))
    np.testing.assert_equal(players[0].get_acceptation_ratio(), 0)
    np.testing.assert_equal(len(players[0]._initial_cards), 8)


def test_game_board_eq():
    board1 = _game.GameBoard([(1, "9♦")], [(1, 80, "♦")])
    board2 = _game.GameBoard([(1, "9d")], [(1, 80, "♦")])
    board3 = _game.GameBoard([(1, "9h")], [(1, 80, "♦")])
    board1.assert_equal(board2)
    np.testing.assert_raises(AssertionError, board1.assert_equal, board3)


def test_board_dump_and_load():
    cards = [v + s for v in _deck.VALUES for s in _deck.SUITS]
    board1 = _game.GameBoard([(k % 4, c) for k, c in enumerate(cards)], [(1, 80, "♦")])
    with tempfile.TemporaryDirectory() as tmp:
        filepath = Path(tmp) / "board_dump_and_load_test.json"
        board1.dump(filepath)
        board2 = _game.GameBoard.load(str(filepath))
    board1.assert_equal(board2)


@genty.genty
class GameTests(TestCase):

    actions = tuple(_PLAYED_CARDS)

    @genty.genty_dataset(
        correct=("❤", "", []),
        incorrect_trump=("♦", "Wrong player for round #1", []),
        unauthorized=("❤", 'Unauthorized Card("K♦") played by player 3', [(3, (3, 'K♦')), (4, (3, 'J❤'))]),
        duplicate=("❤", "Some cards are repeated", [(4, (3, 'J❤'))]),
    )
    def test_gameboard_assert_valid(self, trump_suit, expected, changes):
        actions = list(self.actions)  # duplicate the list
        for index, value in changes:
            actions[index] = value
        board = _game.GameBoard(actions, [(0, 80, trump_suit)])
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
        first_of_finished_2=(32, [28, 29, 30, 31]),
        second_of_second=(5, [4]),
    )
    def test_get_last_round(self, index, expected_inds):
        board = _game.GameBoard(self.actions[:index], [(0, 80, "❤")])
        expected = _deck.CardList([self.actions[i][1] for i in expected_inds], "❤")
        output = board.get_current_round_cards()
        output.assert_equal(expected)

    @genty.genty_repeat(12)
    def test_random_game(self):
        np.random.seed(None)
        board = play_a_game(verbose=True)
        board.assert_valid()
        points = board.points
        assert points.sum() in [162, 182], "Impossible total {}.".format(points.sum())
        np.testing.assert_array_equal(board.points, points, err_msg="Points computation repeats after finishing")
        # dump, reload and check points
        with tempfile.TemporaryDirectory() as tmp:
            filepath = Path(tmp) / "random_board_dump_and_load_test.json"
            board.dump(filepath)
            board2 = _game.GameBoard.load(str(filepath))
        np.testing.assert_array_equal(board2.points, points, err_msg="Points computation is erroneous after reloading")
        np.testing.assert_array_equal(board2.points, points, err_msg="Points computation repeats after finishing")
