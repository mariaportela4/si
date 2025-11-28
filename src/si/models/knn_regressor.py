import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor (Model):
    def __init__(self, k: int=3, distance = euclidean_distance):
        """
        Iniatialize the KNN Regressor.

        Parameters
        ----------
        k: int. Default = 3
            Number of nearest neighbors to consider for prediction.
        distance: callable. Default = euclidean distance
            Function to compute the distance between to points.
        """
        self.k = k # Number of neighbors
        self.distance = distance # Distance metric
        self.train_data = None # Placeholder for training data

    def _fit(self, dataset:Dataset):
        """
        Store the treining datat for future predictions.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model on.
        """
        self.train_data = dataset 

    def _get_neighbors(self, sample:np.ndarray) -> np.ndarray:
        """
        Find the k-nearest neighbors for a given sample.

        Parameters
        ----------
        sample: np.ndarray
            A single sample (1D array) for which to find nighbors.
        Returns
        -------
        np.ndarray
            The target values (y) of the k-nearest neighbors.
        """
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)
        
        # Distances between the sample and all training
        distances = np.array([self.distance(sample, train_sample.reshape(1,-1)) for train_sample in self.train_data.X])
        # Get indices of the k smallest distances
        k_indices = np.argsort(distances)[:self.k]
        # Target values of the k-nearest neighbors
        return self.train_data.y[k_indices]
    
    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict target values for all samples in the dataset.

        Parameters
        ----------
        dataset: Dataset 
            The dataset for which to make predictions.
        Returns
        -------
        np.ndarray
            Predicted target values for each sample.
        """

    # For each sample, get neighbors and predict as the mean of their target values
        predictions = np.array([np.mean(self._get_neighbors(sample)) for sample in dataset.X])
        return predictions

    def _score(self, dataset:Dataset) -> float:
        """
        Compute the Root Mean Squared Error (RMSE) for the predictions.

        Parameters
        ----------
        dataset: Dataset
            The dataset for which to compute the score
        Returns
        -------
        float
            RMSE between predictions and true target values
        """
        # Generate predictions and compute RMSE
        predictions = self._predict(dataset)
        return rmse(dataset.y, predictions)
