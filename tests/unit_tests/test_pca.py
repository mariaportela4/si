import os
from unittest import TestCase
from si.io.csv_file import read_csv
from si.decomposition.pca import PCA

class TestPCA(TestCase):
    def setUp(self):
        self.csv_file = os.path.join('datasets', 'iris', 'iris.csv')
        self.dataset = read_csv(filename=self.csv_file, features=True, label=True)

    def test_fit(self):
        estimator = PCA(n_components=2)
        estimator._fit(self.dataset)  

        self.assertEqual(estimator.n_components, 2)
        self.assertEqual(estimator.components.shape[0], 2)  
        self.assertEqual(estimator.components.shape[1], self.dataset.X.shape[1])  
        self.assertEqual(len(estimator.explained_variance), 2) 
        self.assertTrue(estimator.fitted)  

    def test_transform(self):
        
        estimator = PCA(n_components=2)
        estimator._fit(self.dataset)
        new_dataset = estimator._transform(self.dataset)

        self.assertEqual(new_dataset.X.shape[1], 2)  
        self.assertEqual(new_dataset.X.shape[0], self.dataset.X.shape[0])  