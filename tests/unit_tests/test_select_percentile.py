import os
from unittest import TestCase
from datasets import DATASETS_PATH
from si.feature_selection.select_percentile import SelectPercentile
from si.io.csv_file import read_csv
from si.statistics.f_classification import f_classification


class TestSelectPercentile(TestCase):
    def setUp(self):
        self.csv_file= os.path.join(DATASETS_PATH, 'iris','iris.csv')
        self.dataset= read_csv(filename=self.csv_file,features=True, label=True)
    
    def test_fit(self):
        select_percentile= SelectPercentile(score_func=f_classification, percentile=40)
        select_percentile.fit(self.dataset)

        self.assertEqual(select_percentile.F.shape[0], self.dataset.X.shape[1])
        self.assertEqual(select_percentile.p.shape[0], self.dataset.X.shape[1])

    def test_transform(self):
        select_percentile = SelectPercentile(score_func=f_classification, percentile=40)
        select_percentile.fit(self.dataset)
        new_dataset= select_percentile.transform(self.dataset)

        expected_features= int(len(self.dataset.features)*0.4)
        self.assertEqual(len(new_dataset.features),expected_features)
        self.assertEqual(new_dataset.X.shape[1], expected_features)       