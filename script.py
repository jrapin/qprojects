# pylint: disable-all
import numpy as np
import qprojects
import keras


network = qprojects.BasicNetwork(model_filepath="split_playability_52.h5", verbose=0, learning_rate=0.01)
#network = qprojects.BasicNetwork(model_filepath=None, verbose=0, learning_rate=0.01)
network.online_training = False
players = [qprojects.IntelligentPlayer(network) for _ in range(4)]

#learning_rate = 0.000001
learning_rate = 0.0001
#optimizer = keras.optimizers.SGD(lr=learning_rate, nesterov=False)
optimizer = keras.optimizers.RMSprop(lr=learning_rate, clipnorm=1)
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
index = np.random.randint(len(network._queue))
data = network._queue._data[index]
print(data[1])
output = network.predict(data[0])
output = network.model.predict(data[0][None, :, :])[0, :, :]
#print(np.round(output[:, None], 1))
np.concatenate([data[1], np.round(output, 1)], axis=1)

network._queue.get_random_selection(15)

learning_rate = 0.001
#optimizer = keras.optimizers.SGD(lr=learning_rate, nesterov=False)
optimizer = keras.optimizers.RMSprop(lr=learning_rate)  # , clipnorm=1)
network.model.compile(loss=network.output_framework.error, optimizer=optimizer)


num_external_iter = 0
batch_size = 16
for p in players:
    p.reinitialize()
while True:
    for k in range(50):
        if not (k + 1) % 10:
            print(num_external_iter + 1, k + 1)
        board = play_a_game(players, verbose=False)
    print("Acceptation ratio: {}".format(players[0].get_instantaneous_acceptation_ratio()))
    num_external_iter += 1
    #
    policy = training_policy(max_num_iter=10)
    cond = next(policy)  # prime the policy
    while cond:
        output = network.fit(epochs=1, batch_size=batch_size)
        cond = policy.send((output.history['loss'][0], output.history['val_loss'][0]))


def training_policy(max_num_iter=10):
    prev_val_loss = None
    loss, val_loss = yield True
    assert isinstance(loss, (int, float)), "Wrong type for loss {}: {}".format(loss, type(loss))
    assert isinstance(val_loss, (int, float)), "Wrong type for loss {}: {}".format(val_loss, type(val_loss))
    for k in range(max_num_iter - 1):
        continue_iterations = False
        if prev_val_loss is not None and prev_val_loss > val_loss:
            print("val loss decreases")
            continue_iterations = True
        elif val_loss < loss:
            print("val loss lower than loss")
            continue_iterations = True
        prev_val_loss = val_loss
        loss, val_loss = yield continue_iterations
    yield False



# network.model.save("penalty_71.h5")
network.model.save("split_playability_85.h5")


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
# TODO: look at outputs
# TODO: offline learning
# TODO: add random composant
# TODO: split model
