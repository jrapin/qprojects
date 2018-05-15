import numpy as np
from . import _deck


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


def test_card_points():
    card = _deck.Card(*"J♦")
    np.testing.assert_equal(card.points("♦"), 20)
    np.testing.assert_equal(card.points("❤"), 2)


def test_game_initialization():
    np.random.seed(12)
    game = _deck.Game([_deck.DefaultPlayer() for _ in range(4)])
    # check no duplicate
    playable = {c for p in game.players for c in p.cards}
    assert len(playable) == 32
    

