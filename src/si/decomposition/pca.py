import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset

class PCA (Transformer):
    def __init__(self, n_components:int, **kwargs):
        """
        Principal Component Analysis (PCA)

        Parameters
        ----------
        n_components: int
            Number of principal components to retain.
        """
        super().__init__(**kwargs)
        self.n_components = n_components
        self.fitted = False
        self.mean = None
        self.components = None
        self.explained_variance = None

    def _fit(self, dataset:Dataset) -> 'PCA':
        """
        Fit the PCA model using the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset used to train the PCA model.
        
        Returns
        -------
        PCA
            The trained PCA instance.
        """
        # Validate the number of components
        if not 0 < self.n_components <= dataset.shape()[1]:
            raise ValueError('n_components must be a positive integer no greater than the number of features.')
        
        # Center the dataset
        self.mean = np.mean(dataset.X, axis=0)
        centered_data = dataset.X - self.mean

        # Calculate covariance matrix and perform eigendecomposition
        covariance_matrix = np.cov(centered_data, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        #Selection of the top n_components based on eigenvalues
        sorted_index = np.argsort(eigenvalues)[::-1][:self.n_components]
        self.components = eigenvectors[:, sorted_index].T
        self.explained_variance = eigenvalues[sorted_index] / np.sum(eigenvalues)

        self.fitted = True
        return self
    
    def _transform(self, dataset:Dataset) -> Dataset:
        """
        Transfrom the dataset using the fitted principak components.

        Parameters
        ---------
        dataset: Dataset
            The dataset to trasnsform. Must have the same features as the training dataset.

        Returns
        -------
        Dataset
            A new dataset with reduced dimensions containing the projected features
        """
        if not self.fitted:
            raise ValueError('PCA model must be fitted before calling _transform')
        
        # Center the data using the training mean
        centered_data = dataset.X -self.mean

        # Project the data onto the principal components
        reduced_data = np.dot(centered_data, self.components.T)

        # Create a new dataset with the reduced features
        feature_names = [f"PC{i+1}" for i in range(self.n_components)]
        return Dataset(X = reduced_data, y = dataset.y, features = feature_names, label = dataset.label)
