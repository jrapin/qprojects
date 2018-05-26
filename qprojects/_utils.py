#-*- coding: utf-8 -*

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
