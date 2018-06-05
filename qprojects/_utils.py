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
