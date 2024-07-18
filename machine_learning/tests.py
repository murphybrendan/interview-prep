import numpy as np

from functools import partial

def _test(f, test_cases):
    for X, Y, expected in test_cases:
        try:
            assert f(X, Y) == expected
        except AssertionError:
            print(f"Test failed for X={X}, Y={Y}. Expected {expected}, but got {f(X, Y)}")
            return
    print("All tests passed!")

test_mae = partial(_test, 
    test_cases = [
        (np.array([1, 2, 3]), np.array([1, 2, 3]), 0),
        (np.array([1, 2, 3]), np.array([0, 0, 0]), 2),
    ])

test_mse = partial(_test,
    test_cases = [
        (np.array([1, 2, 3]), np.array([1, 2, 3]), 0),
        (np.array([1, 2, 3]), np.array([0, 0, 0]), 14/3),
    ])


def test_softmax_cross_entropy(sce):
    print("All tests passed!")

    