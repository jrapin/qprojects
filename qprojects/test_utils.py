# -*- coding: utf-8 -*
from unittest import TestCase
import genty
import numpy as np
from . import _utils


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
