import numpy as np
from si.base.estimator import Estimator
from si.base.transformer import Transformer
from si.data.dataset import Dataset

class VarianceThreshold (Transformer):
    def __init__(self, threshold:float = 0.4, **kwargs):
        self.threshold = threshold
        self.variance = None

    def _fit(self, dataset: Dataset) -> 'VarianceThreshold':
        self.variance = np.var(dataset.X, axis=0)
    
    def _transform(self, dataset:Dataset) -> Dataset:
        mask = self.variance >= self.threshold
        X = dataset.X[:, mask]
        features = np.array(dataset.features)[mask]

        return Dataset(X=X, features=features, y=dataset.y, label=dataset.label)



