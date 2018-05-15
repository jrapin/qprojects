from . import _deck

def test_game_initialization():
    game = Game([_deck.DefaultPlayer() for _ in range(4)])
    # check no duplicate
    playable = {c for p in game.players for c in p.cards}
    assert len(playable) == 32
    

