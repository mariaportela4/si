import numpy as np
from si.base.model import Model 
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class StackingClassifier (Model):
    """
    StackingClassifier implements a stacking ensemble method for classification.

    The model combines multiple base classifiers whose predictions are used
    as input features for a final (meta) classifier.
    """
    def __init__(self, models, meta_model):
        """
            Initializes the StackingClassifier.

        Parameters
        ----------
        models : list
            List of base models. Each model must implement fit() and predict().
        
        meta_model : Model
            The meta-model responsible for producing the final predictions.
            Must implement fit() and predict().
        """
   
        self.models = models
        self.meta_model = meta_model


    def _fit(self, dataset: Dataset):
        """
        Fits the stacking classifier.

        Steps:
        1. Train all base models on the original dataset.
        2. Get predictions from the initial set of models
        3. Train the final model with the predictions of the initial set of models.

        Parameters
        ----------
        dataset : Dataset
            Training dataset.

        Returns
        -------
        self
            Fitted StackingClassifier instance.
        """
        # 1. train base models
        for m in self.models:
            m.fit(dataset)

        #2. predictions from the initial set of models
        predictions = []
        for model in self.models:
            pred = model.predict(dataset)
            predictions.append(pred.reshape(-1, 1))

        # Combine predictions horizontally
        stacked_features = np.hstack(predictions)

        #3. train final model
        stacked_dataset = Dataset(X=stacked_features, y=dataset.y)
        self.meta_model.fit(stacked_dataset)


        return self
    
    def _predict(self, dataset:Dataset):
        """
        Predicts class labels using the stacking classifier.

        Parameters
        ----------
        dataset : Dataset
            Dataset to predict.

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        predictions = []
        for model in self.models:
            pred = model.predict(dataset)
            predictions.append(pred.reshape(-1, 1))  

        
        stacked_features = np.hstack(predictions)

        
        final_predictions = self.meta_model.predict(Dataset(X=stacked_features, y=None))

        return final_predictions

    def _score(self, dataset:Dataset, predictions: np.ndarray) -> float:
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        dataset : Dataset
            The test data.
        predictions: np.ndarray
            Predictions

        Returns
        -------
        score : float
            Mean accuracy
        """
        return accuracy(dataset.y, predictions)

