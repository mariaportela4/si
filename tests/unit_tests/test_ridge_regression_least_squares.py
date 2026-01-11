from unittest import TestCase
import numpy as np
import os

from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.ridge_regression_least_squares import RidgeRegressionLeastSquares

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


class TestRidgeRegressionLeastSquares(TestCase):

    def setUp(self):
        self.csv_file = os.path.join(DATASETS_PATH, 'cpu', 'cpu.csv')
        self.dataset = read_csv(
            filename=self.csv_file,
            features=True,
            label=True
        )
        self.train_dataset, self.test_dataset = train_test_split(self.dataset)

    def test_fit(self):
        """Test that fit correctly initializes parameters."""
        model = RidgeRegressionLeastSquares()
        model.fit(self.train_dataset)

        self.assertIsNotNone(model.theta)
        self.assertEqual(model.theta.shape[0], self.train_dataset.X.shape[1])

        self.assertIsNotNone(model.theta_zero)

        self.assertIsNotNone(model.mean)
        self.assertIsNotNone(model.std)
        self.assertEqual(len(model.mean), self.train_dataset.X.shape[1])
        self.assertEqual(len(model.std), self.train_dataset.X.shape[1])

    def test_predict(self):
        """Test prediction shape."""
        model = RidgeRegressionLeastSquares()
        model.fit(self.train_dataset)

        predictions = model.predict(self.test_dataset)
        self.assertEqual(predictions.shape[0], self.test_dataset.X.shape[0])

    def test_score(self):
        """Test that score returns a valid MSE."""
        model = RidgeRegressionLeastSquares()
        model.fit(self.train_dataset)

        mse_value = model.score(self.test_dataset)
        self.assertIsInstance(mse_value, float)
        self.assertGreaterEqual(mse_value, 0)

    def test_sklearn_comparison(self):
        """Compare against scikit-learn Ridge with explicit scaling."""
        l2 = 1.0

        model = RidgeRegressionLeastSquares(l2_penalty=l2, scale=True)
        model.fit(self.train_dataset)
        our_mse = model.score(self.test_dataset)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.train_dataset.X)
        X_test_scaled = scaler.transform(self.test_dataset.X)

        sklearn_ridge = Ridge(alpha=l2)
        sklearn_ridge.fit(X_train_scaled, self.train_dataset.y)

        sklearn_predictions = sklearn_ridge.predict(X_test_scaled)
        sklearn_mse = mean_squared_error(
            self.test_dataset.y,
            sklearn_predictions
        )

        self.assertAlmostEqual(our_mse, sklearn_mse, places=4)

        np.testing.assert_allclose(
            model.theta,
            sklearn_ridge.coef_,
            rtol=1e-4
        )

        self.assertAlmostEqual(
            model.theta_zero,
            sklearn_ridge.intercept_,
            places=4
        )
