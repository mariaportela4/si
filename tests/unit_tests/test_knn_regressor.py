import os
from unittest import TestCase

import numpy as np
from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.metrics.rmse import rmse
from si.model_selection import train_test_split
from si.models.knn_regressor import KNNRegressor
from si.statistics.euclidean_distance import euclidean_distance


class TestKNNRegresssor(TestCase):
    def setUp(self):
        self.csv_file= os.path.join(DATASETS_PATH, 'cpu','cpu.csv')
        self.dataset= read_csv(filename=self.csv_file,features=True, label=True)

    def test_fit(self):
        knn = KNNRegressor(k=4)
        knn._fit(self.dataset)

        self.assertTrue(knn.train_data is not None)
        self.assertTrue(np.all(self.dataset.X == knn.train_data.X))
        self.assertTrue(np.all(self.dataset.y==knn.train_data.y))

    def test_get_neighbors(self):
        knn = KNNRegressor(k=4,distance=euclidean_distance)
        knn._fit(self.dataset)
        sample = self.dataset.X[0]
        neighbors = knn._get_neighbors(sample)
        for neighbor_value in neighbors:
            self.assertIn(neighbor_value, self.dataset.y)
    
    def test_predict(self):
        knn = KNNRegressor(k=4)
        train_dataset, test_dataset = train_test_split(self.dataset, test_size=0.2, random_state=45)
        knn._fit(train_dataset)
        predictions = knn._predict(test_dataset)

        self.assertEqual(predictions.shape[0], test_dataset.X.shape[0])
        self.assertTrue(np.all(predictions>=np.min(train_dataset.y)))
        self.assertTrue(np.all(predictions<=np.max(train_dataset.y)))

    def test_score(self):
        knn = KNNRegressor(k=4)
        train_dataset, test_dataset = train_test_split(self.dataset, test_size=0.2,random_state=45)
        knn._fit(train_dataset)
        score=knn._score(test_dataset)

        self.assertGreater(score,0)

        predictions=knn._predict(test_dataset)
        expected_rmse=rmse(test_dataset.y,predictions)
        self.assertAlmostEqual(score,expected_rmse,places=5)
        

        