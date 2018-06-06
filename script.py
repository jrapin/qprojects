# pylint: disable-all
import numpy as np
import qprojects


network = qprojects.BasicNetwork(model_filepath="basic_network_last_89.h5", verbose=0, learning_rate=0.00001)
players = [qprojects.NetworkPlayer(network) for _ in range(4)]


def play_a_game(players, verbose=False):
    qprojects.initialize_players_cards(players)
    board = qprojects.GameBoard([], [(0, 80, np.random.choice([l for l in "hdsc"]))])
    qprojects.play_game(board, players, verbose=verbose)
    return board

# for p in players:
#    p.reinitialize()
# qprojects.initialize_players_cards(players)
#board = qprojects.GameBoard([], [(0, 80, np.random.choice([l for l in "hdsc"]))])
# players[3]._get_expectations(board)


k = 0
for p in players:
    p.reinitialize()
while True:
    k = k + 1
    print("#{}".format(k))
    board = play_a_game(players, verbose=False)
    print("Acceptation ratio: {}".format(players[0].get_instantaneous_acceptation_ratio()))


index = 8
data = network._queue._data[index]
print(data[1])
output = network.predict(data[0])[:, None]
print(output)

network._queue.get_random_selection(15)


# network._model.save("basic_network_last_89.h5")


# TODO: look at cost function (verbose=1)
# TODO: predict playable
# TODO: look at outputs
# TODO: offline learning
