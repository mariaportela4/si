import numpy as np

def tanimoto_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the Tanimoto similarity between a single binary samples X and and each binary sample in y.

    Parameters
    ----------
    x (array-like): A single binary sample (1D array or list)
    y (array-like): Multiples binary samples (2D array or list of lists)

    Returns
    -------
    numpy.ndarray: Array of Tanimoto similarities between x and each sample in y.
    """

    # Calculate the dot product of x with every point in y.
    dot_products = np.dot(y,x)

    # Sum of the squares for x and each point in y
    sum_squares_x = np.sum(x**2)
    sum_squares_y = np.sum(y**2, axis=1)

    # Sum of the denominator
    denominator = sum_squares_x + sum_squares_y - dot_products

    # Tanimoto Similarity
    similarity = dot_products / denominator

    return similarity