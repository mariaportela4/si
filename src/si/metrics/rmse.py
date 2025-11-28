import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Root Mean Squared Error (RMSE) between true and predicted values.

    Parameters
    ----------
    y_true: numpy.ndarray or list 
        Actual target values.
    y_pred: numpy.ndarray or list
        Predicted target values
    
    Returns 
    -------
    float
        The RMSE, defined as the square root of the mean squared difference between y_true and y_pred.
    """
    return np.sqrt(np.sum((y_true - y_pred) ** 2) / len(y_true))