import os
from unittest import TestCase

import numpy as np

from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.model_selection.split import stratified_train_test_split


class TestStratifiedSplit(TestCase):
    def setUp(self):
        self.csv_file= os.path.join(DATASETS_PATH, 'iris','iris.csv')
        self.dataset= read_csv(filename=self.csv_file,features=True, label=True)
    
    def test_stratified(self):
        train, test = stratified_train_test_split(self.dataset, test_size=0.2,random_state=145)

        total_samples = self.dataset.X.shape[0]
        expected_test_size= int(total_samples*0.2)
        expected_train_size = total_samples - expected_test_size

        self.assertEqual(test.X.shape[0], expected_test_size)
        self.assertEqual(train.X.shape[0], expected_train_size)

        unique_original, counts_original=np.unique(self.dataset.y,return_counts=True)
        unique_train, counts_train=np.unique(train.y,return_counts=True)
        unique_test,counts_test=np.unique(test.y,return_counts=True)

        for i, label in enumerate(unique_original):
            original_ratio=counts_original[i]/total_samples
            train_ratio=counts_train[i]/train.X.shape[0]
            test_ratio=counts_test[i]/test.X.shape[0]

            self.assertAlmostEqual(original_ratio,train_ratio,places=1)
            self.assertAlmostEqual(original_ratio,test_ratio,places=1)

    def test_stratifies_reproducibility(self):
        train1, test1=stratified_train_test_split(self.dataset,test_size=0.2,random_state=10)
        train2, test2=stratified_train_test_split(self.dataset,test_size=0.2,random_state=10)
        train3, test3=stratified_train_test_split(self.dataset,test_size=0.2,random_state=15)
        
        self.assertTrue(np.array_equal(train1.X,train2.X))
        self.assertTrue(np.array_equal(train1.y,train2.y))
        self.assertTrue(np.array_equal(test1.X,test2.X))
        self.assertTrue(np.array_equal(test1.y,test2.y))

        self.assertFalse(np.array_equal(train1.X,train3.X))
        self.assertFalse(np.array_equal(test1.X,test3.X))



