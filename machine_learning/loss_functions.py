#%%
import numpy as np

import tests


"""
Implement MAE
"""
def mean_absolute_error(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Calculates the mean absolute error between two vectors X and Y.
    Mean absolute error is the mean of the pairwise absolute difference between X and Y.
    
    Args:
        X: np.ndarray of shape (n,) - the predicted values
        Y: np.ndarray of shape (n,) - the true values
    
    Returns:
        float: the mean absolute error between X and Y
    """
    return np.mean(np.abs(X - Y))

tests.test_mae(mean_absolute_error)

#%%

"""
Implement MSE
"""
def mean_squared_error(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Calculates the mean squared error between two vectors X and Y.
    Mean squared error is the mean of the pairwise squared difference between X and Y.

    Args:
        X: np.ndarray of shape (n,) - the predicted values
        Y: np.ndarray of shape (n,) - the true values
    
    Returns:
        float: the mean squared error between X and Y
    """
    return np.mean((X - Y) ** 2)

tests.test_mse(mean_squared_error)

#%%
"""
Implement softmax cross-entropy
"""