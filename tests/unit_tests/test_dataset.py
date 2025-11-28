import unittest

import numpy as np

from si.data.dataset import Dataset


class TestDataset(unittest.TestCase):

    def test_dataset_construction(self):

        X = np.array([[1, 2, 3], [4, 5, 6]])
        y = np.array([1, 2])

        features = np.array(['a', 'b', 'c'])
        label = 'y'
        dataset = Dataset(X, y, features, label)

        self.assertEqual(2.5, dataset.get_mean()[0])
        self.assertEqual((2, 3), dataset.shape())
        self.assertTrue(dataset.has_label())
        self.assertEqual(1, dataset.get_classes()[0])
        self.assertEqual(2.25, dataset.get_variance()[0])
        self.assertEqual(1, dataset.get_min()[0])
        self.assertEqual(4, dataset.get_max()[0])
        self.assertEqual(2.5, dataset.summary().iloc[0, 0])

    def test_dataset_from_random(self):
        dataset = Dataset.from_random(10, 5, 3, features=['a', 'b', 'c', 'd', 'e'], label='y')
        self.assertEqual((10, 5), dataset.shape())
        self.assertTrue(dataset.has_label())

    def test_dataset_dropna(self):
        # Test filling NaN with a fixed value (0)
        X= np.array([[1,2], [3, np.nan], [4,5], [np.nan, 6]])
        y= np.array([0,1,0,1])

        X_expected= np.array([[1,2],[3,0], [4,5], [0,6]]) # After filled
        dataset= Dataset(X,y)
        dataset_expected= Dataset(X_expected,y)
        dataset.fillna(0)

        # Make sure no NaN values are left
        self.assertFalse(np.isnan(dataset.X).any())

        #Check if the arrays match
        np.testing.assert_array_equal(dataset.X, dataset_expected.X)
        np.testing.assert_array_equal(dataset.y, dataset_expected.y)
       
    def test_fillna_mean(self):
        # Test filling NaN with the mean of each feature
        X= np.array([[1,2],[3,np.nan],[4,5],[np.nan,6]])
        y= np.array([0,1,0,1])

        X_expected= np.array([[1,2],[3,(2+5+6)/3],[4,5],[(1+3+4)/3,6]]) # After filling NaN with the mean
        dataset= Dataset(X,y)
        dataset_expected = Dataset(X_expected, y)
        dataset.fillna('mean')

        self.assertFalse(np.isnan(dataset.X).any())

        np.testing.assert_array_almost_equal(dataset.X, dataset_expected.X, decimal=6)
        np.testing.assert_array_equal(dataset.y, dataset_expected.y)

    def test_fillna_median(self):
        X= np.array([[1,2],[3,np.nan],[4,5],[np.nan,6]])
        y= np.array([0,1,0,1])

        X_expected= np.array([[1,2],[3,5],[4,5],[3,6]]) # Median of [2,5,6] is 5 and of [1,3,4] is 3
        dataset= Dataset(X,y)
        dataset_expected= Dataset(X_expected, y)
        dataset.fillna('median')

        self.assertFalse(np.isnan(dataset.X).any())

        np.testing.assert_array_equal(dataset.X, dataset_expected.X)
        np.testing.assert_array_equal(dataset.y, dataset_expected.y)

    def test_remove_index(self):
        # Test removing a sample by index
        X= np.array([[1,2],[3,4],[5,6],[7,8]])
        y= np.array([0,1,0,1])
        dataset= Dataset(X,y)
        dataset.remove_by_index(1)

        self.assertEqual(len(dataset.X),3)
        self.assertEqual(len(dataset.y),3)

        np.testing.assert_array_equal(dataset.X, np.array([[1,2],[5,6],[7,8]]))
        np.testing.assert_array_equal(dataset.y, np.array([0,0,1]))