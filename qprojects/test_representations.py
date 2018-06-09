import numpy as np
from . import _representations
from . import _game


def test_explicit_representation():
    players = [_game.DefaultPlayer() for _ in range(4)]
    _game.initialize_players_cards(players)
    board = _game.GameBoard([], [(0, 80, "❤")])
    _game.play_game(board, players, verbose=True)
    Repr = _representations.ExplicitRepresentation
    representation1 = Repr.create(board, players[0])
    representation2 = Repr.create(board, players[0], no_last=True)
    np.testing.assert_array_equal(representation1.shape, Repr.shape)
    np.testing.assert_array_equal(representation1[:-1, :], representation2[:-1, :])
    np.testing.assert_equal(np.sum(representation1[-1, :] - representation2[-1, :]), 2)
