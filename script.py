# pylint: disable-all
import numpy as np
import qprojects
import keras


#network = qprojects.BasicNetwork(model_filepath="playability_93.h5", verbose=0, learning_rate=0.01)
network = qprojects.BasicNetwork(model_filepath=None, verbose=0, learning_rate=0.01)
players = [qprojects.IntelligentPlayer(network) for _ in range(4)]

#learning_rate = 0.000001
learning_rate = 0.00001
optimizer = keras.optimizers.SGD(lr=learning_rate, nesterov=False)
optimizer = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
network.model.compile(loss=network.output_framework.error, optimizer=optimizer)


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


index = 16
index = np.random.randint(1000)
data = network._queue._data[index]
print(data[1])
output = network.predict(data[0])
output = network.model.predict(data[0][None, :, :])[0, :, :]
#print(np.round(output[:, None], 1))
np.concatenate([data[1], np.round(output, 1)], axis=1)

network._queue.get_random_selection(15)

learning_rate = 0.00001
optimizer = keras.optimizers.SGD(lr=learning_rate, nesterov=False)
optimizer = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
network.model.compile(loss=network.output_framework.error, optimizer=optimizer)

network.fit(batch_size=8)

network.model.save("penalty_71.h5")
# network.model.save("playability_93.h5")


errors = [(x, y) for x, y in network._queue._data if np.max(abs(y[:, 0] - network.model.predict(x[None, :, :])[0, :, 0])) > 0.7]
print(len(errors))


index = np.random.randint(len(errors))
data = errors[index]
print(data[1])
output = network.predict(data[0])
output = network.model.predict(data[0][None, :, :])[0, :, :]
#print(np.round(output[:, None], 1))
np.concatenate([data[1], np.round(output, 1)], axis=1)

batch_data = errors
batch_representation, batch_expected = (np.array(x) for x in zip(*batch_data))
X = np.array(batch_representation)
y = np.array(batch_expected)
thresh = int(0.9 * X.shape[0])
X_train, y_train = [x[:thresh, :, :] for x in (X, y)]
X_test, y_test = [x[thresh:, :, :] for x in (X, y)]
epochs = 100
shuffle = True
batch_size = 4
network.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=shuffle, validation_data=(X_test, y_test))


# TODO: look at cost function (verbose=1)
# TODO: predict playable
# TODO: look at outputs
# TODO: offline learning
# TODO: add random composant
