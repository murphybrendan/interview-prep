import numpy as np
import torch

from functools import partial


def _test(f, test_cases):
    failures = []
    for X, Y, expected in test_cases:
        try:
            assert torch.allclose(f(X, Y), torch.tensor(expected), rtol=1e-5, atol=1e-4)
        except AssertionError:
            failures.append(f"Test failed for X={X}, Y={Y}. Expected {expected}, but got {f(X, Y)}")
    if failures:
        print('\n'.join(failures))
    else:
        print("All tests passed!")


test_mae = partial(_test, 
    test_cases = [
        (torch.tensor([1., 2., 3.]), torch.tensor([1, 2, 3]), 0.),
        (torch.tensor([1., 2., 3.]), torch.tensor([0, 0, 0]), 2.),
    ])


test_mse = partial(_test,
    test_cases = [
        # Existing cases
        (torch.tensor([1., 2., 3.]), torch.tensor([1., 2., 3.]), 0.),
        (torch.tensor([1., 2., 3.]), torch.tensor([0., 0., 0.]), 14/3),
        
        # Additional cases
        # Small error case
        (torch.tensor([1., 2., 3.]), torch.tensor([1.1, 2.1, 2.9]), 0.01),
        
        # Large error case
        (torch.tensor([1., 2., 3.]), torch.tensor([10., 20., 30.]), 378.0),
        
        # Negative values
        (torch.tensor([-1., -2., -3.]), torch.tensor([1., 2., 3.]), 18.6667),
        
        # Mixed positive and negative
        (torch.tensor([-1., 0., 1.]), torch.tensor([1., 0., -1.]), 2.6667),
        
        # Single value
        (torch.tensor([5.]), torch.tensor([10.]), 25.),
        
        # 2D tensor (multiple samples)
        (torch.tensor([[1., 2.], [3., 4.]]), torch.tensor([[1.5, 2.5], [3.5, 4.5]]), 0.25),
        
        # Zero prediction for non-zero target
        (torch.tensor([0., 0., 0.]), torch.tensor([1., 2., 3.]), 14/3),
        
        # Very large numbers
        (torch.tensor([1e6, 2e6]), torch.tensor([1.1e6, 2.1e6]), 1e10),
        
        # Very small numbers
        (torch.tensor([1e-6, 2e-6]), torch.tensor([1.1e-6, 2.1e-6]), 2e-14),
    ])


test_bcel = partial(_test,
    test_cases = [
        # Existing cases
        (torch.tensor([1., 0.]), torch.tensor([1., 0.]), 1e-7),
        (torch.tensor([0., 1.]), torch.tensor([1., 0.]), float(-np.log(1e-7))),
        (torch.tensor([0.5, 0.5]), torch.tensor([1., 0.]), float(-np.log(0.5))),
        
        # Additional cases
        # Near-perfect predictions
        (torch.tensor([0.9, 0.1]), torch.tensor([1., 0.]), 0.1054),
        (torch.tensor([0.1, 0.9]), torch.tensor([0., 1.]), 0.1054),
        
        # More realistic predictions
        (torch.tensor([0.7, 0.3]), torch.tensor([1., 0.]), 0.3567),
        (torch.tensor([0.3, 0.7]), torch.tensor([0., 1.]), 0.3567),
        
        # Multi-sample case
        (torch.tensor([[0.7, 0.3], [0.2, 0.8]]), torch.tensor([[1., 0.], [0., 1.]]), 0.2899),
        
        # Edge cases
        (torch.tensor([1e-7, 1-1e-7]), torch.tensor([1., 0.]), 16.0302),
        (torch.tensor([1-1e-7, 1e-7]), torch.tensor([0., 1.]), 16.0302),
        
        # Balanced case
        (torch.tensor([0.5, 0.5]), torch.tensor([0.5, 0.5]), 0.6931),
    ])


test_softmax_cross_entropy = partial(_test,
    test_cases = [
        # 1. Basic functionality
        (torch.tensor([1., 2., 3.]), torch.tensor([0, 1, 0]), 1.4076),
        (torch.tensor([[1., 2., 3.], [4., 5., 6.]]), torch.tensor([[0, 1, 0], [1, 0, 0]]), 1.9076),
        (torch.tensor([1., 2., 3.]), torch.tensor([0.1, 0.7, 0.2]), 1.3076),
        (torch.tensor([[1., 2., 3.], [4., 5., 6.]]), torch.tensor([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4]]), 1.3076),

        # 2. Edge cases
        (torch.tensor([1000., 2000., 3000.]), torch.tensor([0, 0, 1]), 0.0),
        (torch.tensor([1e-7, 2e-7, 3e-7]), torch.tensor([1, 0, 0]), 1.0986),
        (torch.tensor([-1000., 0., 1000.]), torch.tensor([0, 1, 0]), 1000.0),
        (torch.tensor([1., 1., 1.]), torch.tensor([0, 1, 0]), 1.0986),

        # 3. Special cases
        (torch.tensor([0., 10., 0.]), torch.tensor([0, 1, 0]), 9.0796e-05),
        (torch.tensor([10., 0., 0.]), torch.tensor([0, 1, 0]), 10.),
        (torch.tensor([1., 1., 1.]), torch.tensor([1/3, 1/3, 1/3]), 1.0986),
    ])


test_multi_label_cross_entropy = partial(_test,
    test_cases = [
        # Normal cases
        (torch.tensor([0, 1, 2]), torch.tensor([0, 1, 1]), 0.3778),
        (torch.tensor([-1, 0, 1]), torch.tensor([0, 0, 1]), 0.4399),

        # Perfect and worst predictions
        (torch.tensor([1000, -1000]), torch.tensor([1, 0]), 0.0),
        (torch.tensor([-1000., 1000.]), torch.tensor([1, 0]), 1000.),
        
        # Very small inputs
        (torch.tensor([1e-8, -1e-8]), torch.tensor([1, 0]), 0.6931),
        
        # Multiple samples
        (torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 0], [0, 1]]), 1.3767),

        # All positive or all negative predictions
        (torch.tensor([1, 2, 3]), torch.tensor([1, 1, 1]), 0.1629),
        (torch.tensor([-1, -2, -3]), torch.tensor([0, 0, 0]), 0.1629),
    ])

    