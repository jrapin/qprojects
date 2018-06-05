# pylint: disable-all
import numpy as np
import qprojects


network = qprojects.BasicNetwork(model_filepath="basic_network_last.h5", verbose=0)
players = [qprojects.NetworkPlayer(network) for _ in range(4)]
for p in players:
    p.reinitialize()


def play_a_game(players, verbose=False):
    qprojects.initialize_players_cards(players)
    board = qprojects.GameBoard([], [(0, 80, np.random.choice([l for l in "hdsc"]))])
    qprojects.play_game(board, players, verbose=verbose)
    return board


board = play_a_game(players, verbose=False)

for k in range(20000):
    print("#{}".format(k))
    board = play_a_game(players, verbose=False)
    print("Acceptation ratio: {}".format(players[0].get_instantaneous_acceptation_ratio()))

del network


network._queue.get_random_selection(15)
