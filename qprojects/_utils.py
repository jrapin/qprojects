# -*- coding: utf-8 -*
import itertools
import functools


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
        return self._SINGLETONS
