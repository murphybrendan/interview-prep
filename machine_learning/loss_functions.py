#%%
import torch

import tests


"""
Implement MAE
Used for regression
"""
def mean_absolute_error(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Calculates the mean absolute error between two vectors X and Y.
    Mean absolute error is the mean of the pairwise absolute difference between X and Y.
    
    Args:
        X: torch.Tensor of shape (n,) - the predicted values
        Y: torch.Tensor of shape (n,) - the true values
    
    Returns:
        float: the mean absolute error between X and Y
    """
    return torch.mean(torch.abs(X - Y))

tests.test_mae(mean_absolute_error)


#%%
import torch
import tests

"""
Implement MSE
Used for regression
"""
def mean_squared_error(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Calculates the mean squared error between two vectors X and Y.
    Mean squared error is the mean of the pairwise squared difference between X and Y.

    Args:
        X: torch.Tensor of shape (n,) - the predicted values
        Y: torch.Tensor of shape (n,) - the true values
    
    Returns:
        float: the mean squared error between X and Y
    """
    return torch.mean((X - Y) ** 2)

tests.test_mse(mean_squared_error)


#%%
import torch
import tests

"""
Implement binary cross-entropy loss
Used for binary classification
"""
def binary_cross_entropy_loss(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Calculates the mean binary cross entropy loss.
    X is assumed to be a vector of probabilities, where X[i] is the probability of being in the positive class.

    Args:
        X: torch.Tensor of shape (n,) - the predicted values
        Y: torch.Tensor of shape (n,) - the true values
    
    Returns:
        float: the mean squared error between X and Y
    """
    # Clip X and 1-X to be in [1e-7, 1-1e-7] for numerical stability.
    # That way, log(X) and log(1-X) are always defined (never log(0)).
    return -torch.mean(Y * torch.log(torch.clamp(X, 1e-7, 1 - 1e-7)) 
                    + (1 - Y) * torch.log(torch.clamp(1 - X, 1e-7, 1 - 1e-7)))

tests.test_bcel(binary_cross_entropy_loss)


# %%
import torch
import tests

"""
Implement softmax cross-entropy loss
Used for multi-class classification
"""
def softmax_cross_entropy_loss(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Calculates the mean softmax cross entropy loss

    Args:
        X: torch.Tensor of shape (n, C) - the logits for each class
        Y: torch.Tensor of shape (n, C) - the class labels
    
    Returns:
        float: the softmax cross-entropy loss between X and Y
    """
    # Subtract the max for stability before exp for numerical stability
    # keepdim=True for correct broadcasting
    X_shifted = X - torch.max(X, dim=-1, keepdim=True).values

    # Trick of log softmax:
    # log_softmax(X) = log(exp(X) / sum(exp(X))) - log of division = subtraction of logs
    # log_softmax(X) = log(exp(X)) - log(sum(exp(X))) - log(exp(x)) = x
    # log_softmax(X) = X - log(sum(exp(X)))
    # keepdims=True for correct broadcasting
    log_sum_exp = torch.log(torch.sum(torch.exp(X_shifted), dim=-1, keepdim=True))
    log_softmax = X_shifted - log_sum_exp

    # Y is normally one-hot encoded. In this case, the the loss is 0 everywhere except for the log probability of the true label.
    # You could directly index into log_softmax for that.
    # However, Y can also be a probability distribution. In this case, you want to sum the losses, scaled by the true probability distribution.
    return -torch.mean(torch.sum(Y * log_softmax, dim=-1))

tests.test_softmax_cross_entropy(softmax_cross_entropy_loss)


#%%
import torch
import tests

def multi_label_cross_entropy_loss(X: torch.Tensor, Y: torch.Tensor) -> float:
    # Sigmoid of X.
    # Ensures stability for negative numbers (which could overflow np.exp(-X))
    # Ensures stability of extreme values
    log_sigmoid = torch.where(X >= 0,
                              -torch.log1p(torch.exp(-X)),
                              X - torch.log1p(torch.exp(X)))
    
    log_one_minus_sigmoid = torch.where(X >= 0,
                                        -X + log_sigmoid,
                                        -torch.log1p(torch.exp(X)))

    return -torch.mean(Y * log_sigmoid + (1 - Y) * log_one_minus_sigmoid)

tests.test_multi_label_cross_entropy(multi_label_cross_entropy_loss)

# %%
