from typing import Callable

import numpy as np

from si.base.transformer import Transformer
from si.data import dataset
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification

class SelectPercentile (Transformer):
    """
    Selects the top percentile of features based on a scoring function.

    Parameters
    ----------
    score_func: callable
        Function takinga Dataset and returning (scores, pvalues).
    
    percentile: float
        Value between 0 and 100 indicating what percentage of features to keep.

    Attributes
    ----------
    F: array-like, shape (n_features,)
    p: array-like, shape(n_feautures,)
        p-values of each feature.
    """

    def __init__ (self, score_func: Callable = f_classification, percentile: float = 10.0, **kwargs ):
        super().__init__(**kwargs)
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None
    

    def _fit(self, dataset: Dataset) -> 'SelectPercentile':
        """
        Compute features scores (F) and p-values
        
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform (self, dataset: Dataset) -> Dataset:
        """
         Select the top features based on the computed F-scores and the specified percentile.

        Parameters
        ----------
        dataset: Dataset

        Returns
        -------
        Dataset
            Dataset with selected features

        """
        # Compute the number of features to select
        num_features = int(len(dataset.features) * (self.percentile / 100))
        num_features = max(1, num_features)  # Ensure at least one feature is selected

        # Select indices of the top features
        top_indices = np.argsort(self.F)[-num_features:]

       # Subset the dataset
        selected_X = dataset.X[:, top_indices]
        selected_features = [dataset.features[i] for i in top_indices]

        return Dataset(X=selected_X, y=dataset.y, features=selected_features, label=dataset.label)

# ----- Example -----

if __name__ == '__main__':
    f_values = np.array([1.2, 3.4, 2.1, 5.6, 4.3, 5.6, 7.8, 6.5, 5.6, 3.2])
    percentile = 40

    num_features= int(len(f_values)*(percentile/100))
    num_features= max(1, num_features)

    top_indices= np.argsort(f_values)[-num_features:]

    selected_f= f_values[top_indices]

    print('All f-values:', f_values)
    print('Percentile:', percentile)
    print('Number of features:', num_features)
    print('Top features indices:', top_indices)
    print('Selected f-values:', selected_f)

     