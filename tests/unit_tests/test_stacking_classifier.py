import os
from unittest import TestCase

from datasets import DATASETS_PATH
from si.ensemble.stacking_classifier import StackingClassifier
from si.io.data_file import read_data_file
from si.metrics.accuracy import accuracy
from si.model_selection.split import stratified_train_test_split
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression
import numpy as np


class TestStackingClassifier(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        self.dataset = read_data_file(filename=self.csv_file, label=True, sep=",")

        self.train_dataset, self.test_dataset = stratified_train_test_split(
            self.dataset, test_size=0.25
        )

        # base models
        self.knn = KNNClassifier()
        self.log_reg = LogisticRegression()
        self.dec_tree = DecisionTreeClassifier()

        # meta model -> final predictions
        self.meta_knn = KNNClassifier()

        
        self.stacking = StackingClassifier(
            models=[self.knn, self.log_reg, self.dec_tree],
            meta_model=self.meta_knn
        )

    def test_fit(self):
        # just ensure fit runs and implied meta-data dimensions are correct
        self.stacking.fit(self.train_dataset)

        # reconstruct the stacked features as in your _fit
        base_preds = []
        for model in self.stacking.models:
            pred = model.predict(self.train_dataset)
            base_preds.append(pred.reshape(-1, 1))
        stacked_features = np.hstack(base_preds)

        # rows = number of training samples
        self.assertEqual(stacked_features.shape[0], self.train_dataset.shape()[0])
        # cols = number of base models
        self.assertEqual(stacked_features.shape[1], len(self.stacking.models))

    def test_predict(self):
        self.stacking.fit(self.train_dataset)
        predictions = self.stacking.predict(self.test_dataset)

        # predictions is a numpy array -> .shape, Dataset uses shape()
        self.assertEqual(predictions.shape[0], self.test_dataset.shape()[0])

    def test_score(self):
        self.stacking.fit(self.train_dataset)

        accuracy_ = self.stacking.score(self.test_dataset)
        expected_accuracy = accuracy(
            self.test_dataset.y,
            self.stacking.predict(self.test_dataset)
        )

        self.assertEqual(round(accuracy_, 2), round(expected_accuracy, 2))
        print(
            f"\nObtained accuracy: {round(accuracy_, 2)}"
            f"\nExpected accuracy: {round(expected_accuracy, 2)}"
        )
