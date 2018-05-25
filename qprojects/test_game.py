import tempfile
from pathlib import Path
import numpy as np
from . import _game
from . import _deck
from ._deck import Card as C


def test_game_initialization():
    #np.random.seed(12)
    game = _game.Game([_game.DefaultPlayer() for _ in range(4)])
    # check no duplicate
    playable = {c for p in game.players for c in p.cards}
    assert len(playable) == 32


def test_game():
    game = _game.Game([_game.DefaultPlayer() for _ in range(4)])
    game.trump_suit = "❤"
    game.play_game(verbose=True)
    raise Exception


def test_game_board_eq():
    board1 = _game.GameBoard([(1, C(*"9♦"))], [(1, 80, "♦")])
    board2 = _game.GameBoard([(1, C(*"9d"))], [(1, 80, "♦")])
    board3 = _game.GameBoard([(1, C(*"9h"))], [(1, 80, "♦")])
    board1.assert_equal(board2)
    np.testing.assert_raises(AssertionError, board1.assert_equal, board3)


def test_board_dump_and_load():
    cards = [C(v, s) for v in _deck.VALUES for s in _deck.SUITS]
    board1 = _game.GameBoard([(k % 4, c) for k, c in enumerate(cards)], [(1, 80, "♦")])
    with tempfile.TemporaryDirectory() as tmp:
        filepath = Path(tmp) / "board_dump_and_load_test.json"
        board1.dump(filepath)
        board2 = _game.GameBoard.load(str(filepath))
    board1.assert_equal(board2)

