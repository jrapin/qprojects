# -*- coding: utf-8 -*
from unittest import TestCase
import genty
import numpy as np
from . import _utils


@_utils.singleton
class ExampleSingleton:
    """A singleton for testing
    """
    pass


def test_singleton():
    sing1 = ExampleSingleton()
    sing2 = ExampleSingleton()
    np.testing.assert_equal(sing1, sing2)


@genty.genty
class UtilsTests(TestCase):

    @genty.genty_dataset(
        equal=([2, 3, 1], ""),
        missing=((1, 2), ["  - missing element(s): {3}."]),
        additional=((1, 4, 3, 2), ["  - additional element(s): {4}."]),
        both=((1, 2, 4), ["  - additional element(s): {4}.", "  - missing element(s): {3}."]),
    )
    def test_assert_set_equal(self, estimate, message):
        reference = {1, 2, 3}
        try:
            _utils.assert_set_equal(estimate, reference)
        except AssertionError as error:
            if not message:
                raise AssertionError("An error has been raised while it should not.")
            np.testing.assert_equal(error.args[0].split("\n")[1:], message)
        else:
            if message:
                raise AssertionError("An error should have been raised.")

    @genty.genty_dataset(
        list_2_none=(list(range(3)), 2, None, [[0, 1], [2, None]]),
        gen_3_0=(range(4), 3, 0, [[0, 1, 2], [3, 0, 0]]),
    )
    def test_grouper(self, iterator, n, fillvalue, expected):
        output = list(_utils.grouper(iterator, n, fillvalue))
        np.testing.assert_array_equal(output, expected)

    @genty.genty_dataset(
        fail=([(100, 300)], 1),
        lower_val=([(400, 300), (200, 350)], 2),
        lower_val_decrease=([(400, 300), (200, 250), (200, 350)], 3),
        lower_val_decrease_max=([(400, 300), (200, 250), (200, 200), (200, 100)], 4),
    )
    def test_epoch_policy(self, losses, num_expected):
        policy = _utils.epoch_policy(max_epoch=4, verbose=True)
        output = [next(policy)]
        for loss_val_loss in losses:
            output.append(policy.send(loss_val_loss))
        np.testing.assert_equal(output, num_expected * [True] + [False])

    @genty.genty_dataset(
        with_list=([1, 2, 3], [1, 2, 3]),
        with_gen=((x for x in [1, 2, 3]), [1, 2, 3]),
    )
    def test_memory_iterator(self, iterable, expected):
        iterable = [1, 2, 3]
        expected = [1, 2, 3]
        memo = _utils.MemoryIterator(iterable)
        nexts, lasts, lasts2 = [], [], []
        for value in memo:
            nexts.append(value)
            lasts.append(memo.last)
            lasts2.append(memo.last)
        np.testing.assert_array_equal(nexts, expected)
        np.testing.assert_array_equal(lasts, expected)
        np.testing.assert_array_equal(lasts2, expected)


def test_memory_iterator_with_coroutine():

    def coroutine():
        value = yield 12
        for _ in range(2):
            value = yield value

    memo = _utils.MemoryIterator(coroutine())
    np.testing.assert_equal(next(memo), 12)  # priming
    np.testing.assert_equal(memo.last, 12)
    np.testing.assert_equal(memo.send(1), 1)
    np.testing.assert_equal(memo.last, 1)


def test_replay_queue():
    queue = _utils.ReplayQueue(3)
    for k in range(7):
        queue.append((k, k))
    _ = str(queue)
    np.testing.assert_array_equal(queue._data, [(6, 6), (4, 4), (5, 5)])
    np.random.seed(25)
    np.testing.assert_equal(queue.get_random_selection(2), [(4, 4), (5, 5)])
    np.testing.assert_equal(queue.get_random_selection(5), [(6, 6), (4, 4), (5, 5)])
