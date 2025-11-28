from typing import Tuple
import numpy as np
from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple [Dataset, Dataset]:
    """
    Split the dataset into random training and testing sets.

    Parameters
    ----------
    dataset: Dataset
        The dataset to split.
    test_size: float. By default, 20%.
        Proportion of the dataset to include in the test split.
    random_state: int. By default 42.

    Returns
    -------
    Tuple [Dataset, Dataset]
        A tuple containing the training and testing datasets.
    """
    # Validate test_size
    if not (0 < test_size < 1):
        raise ValueError ('test_size must be a float between 0 and 1')
    
    np.random.seed(random_state) # set the random seed from reproducibility
    n_samples = dataset.X.shape[0] # get the total number of samples in the dataset
    n_test = int(n_samples * test_size) # calculate the number of samples for the test set

    # Shuffle indices and split
    indices = np.random.permutation(n_samples) # generate a random permutation of indices
    test_idxs, train_indxs = indices[:n_test], indices[n_test:] # split indices into test and train sets

    return (Dataset(X = dataset.X[train_indxs], y = dataset.y[train_indxs], features = dataset.features, label = dataset.label), 
            Dataset(X = dataset.X[test_idxs], y = dataset.y[test_idxs], features = dataset.features, label = dataset.label,))

def stratified_train_test_split (dataset:Dataset, test_size: float = 0.2,
                                 random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets while preserving class distribution.

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float. By default 20%.
        Proportion of the dataset to include in the test slip.
    random_state: int. By default 42.
        Seed for the random number generator

    Returns
    -------
    Tuple[Dataset, Dataset] 
        A tuple containing the treining and testing datasets.
    """
    if not (0 < test_size < 1):
        raise ValueError('test_size must be a float between 0 and 1.')
    
    rng = np.random.default_rng(random_state) # initialize the random number generator with the provided seed
    unique_classes, class_counts = np.unique(dataset.y, return_counts = True) # get unique classes and their counts in the dataset

    # lists to store indices from train and test sets
    train_indxs = []
    test_indxs = []

    # For each class, split its indices into train and test sets
    for cls, count in zip(unique_classes, class_counts):
        class_idxs = np.where(dataset.y == cls)[0] # get all indices for the current class
        rng.shuffle(class_idxs) # suffle the indices for randomness
    
        test_count = int(count * test_size) # calculate the number of test samples for this class
        test_indxs.extend (class_idxs[:test_count])
        train_indxs.extend (class_idxs[test_count:])
        

    return(Dataset(X = dataset.X[train_indxs], y = dataset.y[train_indxs], features = dataset.features, label = dataset.label), 
           Dataset(X = dataset.X[test_indxs], y =dataset.y[test_indxs], features = dataset.features, label = dataset.label))


