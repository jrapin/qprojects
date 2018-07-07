# -*- coding: utf-8 -*
import itertools
import functools
import numpy as np


def assert_set_equal(estimate, reference):
    """Asserts that both sets are equals, with comprehensive error message.
    This function should only be used in tests.

    Parameters
    ----------
    estimate: iterable
        sequence of elements to compare with the reference set of elements
    reference: iterable
        reference sequence of elements
    """
    estimate, reference = (set(x) for x in [estimate, reference])
    elements = [("additional", estimate - reference), ("missing", reference - estimate)]
    messages = ["  - {} element(s): {}.".format(name, s) for (name, s) in elements if s]
    if messages:
        raise AssertionError("\n".join(["Sets are not equal:"] + messages))


def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks

    Parameters
    ----------
    iterable: iterable
        an iterable to group by batches
    n: int
        the number of elements of each batch
    fillvalue: object
        the value for filling the last batch if the iterable length is not a multiple of n
    """
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    # this code is copied from itertools recipes
    args = [iter(iterable)] * n
    return itertools.zip_longest(fillvalue=fillvalue, *args)


class singleton:
    """Singleton decorator
    """

    _SINGLETONS = {}

    def __init__(self, cls):
        self._cls = cls
        functools.update_wrapper(self, cls)

    def __call__(self, *args, **kwargs):
        if self._cls not in self._SINGLETONS:
            self._SINGLETONS[self._cls] = self._cls(*args, **kwargs)
        return self._SINGLETONS[self._cls]


class ReplayQueue:

    def __init__(self, max_len=100):
        self._data = []
        self._max_len = max_len
        self._index = 0

    def append(self, value):
        if len(self._data) == self._max_len:
            self._data[self._index] = value
            self._index = (self._index + 1) % self._max_len
        else:
            self._data.append(value)

    def __len__(self):
        return len(self._data)

    def get_random_selection(self, size):
        if len(self) < size:
            return list(self._data)
        else:
            # needs to be robust to list of tuples
            indices = np.random.choice(len(self._data), size=size, replace=False)
            return [self._data[i] for i in indices]

    def __repr__(self):
        return "ReplayQueue({}): index {}, data {}".format(self._max_len, self._index, self._data)


def epoch_policy(max_epoch=10, verbose=True):
    """Coroutine serving as policy for the number of epoch to perform before creating more data.
    The coroutine yields True if the validation loss is lower than the loss, or the validation loss
    decreases. It stops after at most max_epoch epochs.

    Parameters
    ----------
    max_epoch: int
        maximum number of epoch to perform
    verbose: bool
        whether to print the reason the training is continuing

    Receives
    --------
    tuple
        a tuple of two floats: (loss, validation_loss)

    Yields
    ------
    bool
        whether to continue the training for another epoch.

    Usage
    -----
    policy = epoch_policy(max_epoch=10)
    cond = next(policy)  # prime the policy
    while cond:
        output = network.fit(epochs=1)
        cond = policy.send((output.history['loss'][0], output.history['val_loss'][0]))

    """
    prev_val_loss = None
    loss, val_loss = yield True
    # check the types at the first round
    assert isinstance(loss, (int, float)), "Wrong type for loss {}: {}".format(loss, type(loss))
    assert isinstance(val_loss, (int, float)), "Wrong type for loss {}: {}".format(val_loss, type(val_loss))
    for _ in range(max_epoch - 1):
        reason = ""
        if prev_val_loss is not None and prev_val_loss > val_loss:
            reason = "Validation loss decreased"
        elif val_loss < loss:
            reason = "Validation loss is lower than loss."
        if loss / val_loss > 4:
            reason = ""  # testing is not representative...
        if reason:
            if verbose:
                print(reason)
            prev_val_loss = val_loss
            loss, val_loss = yield True
        else:
            break
    yield False


class MemoryIterator:
    """Iterator with a memory of last element
    """

    def __init__(self, iterable):
        self._iter = iter(iterable)
        self._last = None

    @property
    def last(self):
        return self._last

    def __next__(self):
        self._last = next(self._iter)
        return self._last

    def send(self, value):
        self._last = self._iter.send(value)
        return self._last

    def __iter__(self):
        return self
