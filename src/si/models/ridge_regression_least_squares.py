import numpy as np
from si.base.model import Model
from si.metrics.mse import mse


class RidgeRegressionLeastSquares (Model):
    """
    Ridge Regression based on the closed-form (normal equation) solution.
    This method extends linear regression by incorporating L2 regularization.
    """
    def __init__(self,l2_penalty=1, scale=True,**kwargs):
        super().__init__(**kwargs)
        self.l2_penalty= l2_penalty
        self.scale= scale
        self.theta= None
        self.theta_zero= None
        self.mean= None
        self.std= None
    
    def _fit(self, dataset):
        X = dataset.X
        if self.scale:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            X = (X-self.mean)/self.std
        m,n = X.shape    
        
        X_with_intercept = np.c_[np.ones(m), X] # Add column of ones for the intercept
        
        # Regularization matrix
        regularization= self.l2_penalty * np.eye(n+1)
        regularization[0,0] = 0

        # Solve for theta using the normal equation
        theta= np.linalg.inv(X_with_intercept.T @ X_with_intercept + regularization) @ X_with_intercept.T @ dataset.y

        # Split intercept and coefficients
        self.theta_zero= theta[0]
        self.theta= theta[1:]

        return self
    
    def _predict(self, dataset):
        """
        Predict the output for the dataset
        """
        X = dataset.X
        if self.scale:
            X= (X-self.mean)/self.std
        return np.dot(X, self.theta) + self.theta_zero
    
    def _score(self, dataset, predictions=None):
        """ Compute the Mean Squared Error (MSE) for the dataset.
        """
        predictions= self.predict(dataset)
        return mse (dataset.y, predictions)