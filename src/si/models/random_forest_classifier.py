import numpy as np
from si.base.model import Model
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.models.decision_tree_classifier import DecisionTreeClassifier


class RandomForestClassifier(Model):
    """
    RandomForestClassifier is an ensemple model that combines multiple decision trees to improve accuracy, robustness, and reduce overfitting.
    """
    def __init__(self, n_estimators = 10, max_features = None, min_sample_split = 2, max_depth = 10, mode='gini', seed = None, **kwargs):
        """
        Initialize the RandomForestClassifier.

        Parameters
        ----------
        n_estimators: int. Default = 10
            Number of decision trees in the ensemble.
        max_features: int. Default = None
            Maximum number of features to consider for each tree.
            If none, defaults to sqrt(n_features)
        min_sample_split: int. Default = 2
            Minimum number of samples required to split a node.
        max_depth: int. Default =10
            Maximum depth of each decision tree.
        mode: str. Default  = 'gini'
            Impurity calculation mode.
        seed: int. Default = None
            Seed for random number generation for reproducibility.
        """
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        self.trees = [] # List to store trained treed and their indices

    def _fit(self, dataset: Dataset): 
        """
        Train the RandomForestClassifier using bootstrap samples from the dataset.
        
        Parameters
        ----------
        dataset: Dataset
            Dataset for training.
        
        Return
        ------
        self.RandomForestClassifier
            The training model.
        """

        # Random seed for reproducibility
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Max_features to sqrt(n_features) if not provided
        n_samples, n_features = dataset.X.shape
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))
        
        # Train each decision tree on a botstrap sample
        for _ in range (self.n_estimators):
            # Create bootstrap sample indices (with replacement)
            bootstrap_indices = np.random.choice(n_samples, size= n_samples, replace= True)
            # Randomly select features for the current tree
            feature_indices = np.random.choice(n_features, size= self.max_features, replace= False)
            # Create bootstrap dataset
            bootstrap_X = dataset.X[bootstrap_indices][:, feature_indices]
            bootstrap_y = dataset.y[bootstrap_indices]
            # Train decision tree
            tree = DecisionTreeClassifier(min_sample_split= self.min_sample_split, max_depth = self.max_depth, mode= self.mode)
            tree.fit(Dataset(X=bootstrap_X, y=bootstrap_y))
            # Store the trained tree and its feature indices
            self.trees.append((feature_indices, tree))

        return self
    def _predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the labels for the dataset using majority voting from all trees.

        Parameters
        ----------
        dataset: Dataset
            Dataset for prediction.
        Returns
        -------
        predictions : np.ndarray
            Predicted labels for the dataset.
      """
        tree_predictions = []
        # Get predictions from each tree
        for feature_indices, tree in self.trees:
            # Only the features used by current tree
            selected_features= dataset.X[:, feature_indices]
            sub_dataset= Dataset(X=selected_features, y=None)
            # Predictions from the current tree
            predictions= tree.predict(sub_dataset)
            tree_predictions.append(predictions)
        
        # Transpose 
        tree_predictions = np.array(tree_predictions).T
        final_predictions= []

        # For each sample, get the majority vote across all trees
        for row in tree_predictions:
            # Get unique laels and their counts
            unique, counts = np.unique(row, return_counts = True)
            majority_vote = unique[np.argmax(counts)]
            final_predictions.append(majority_vote)
        return np.array(final_predictions)
    
    def _score(self, dataset: Dataset, predictions: np.ndarray) -> float:
        """
        Evaluate the accuracy of the model on the dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset for evaluation.
        predictions: np.ndarray
            Predicted labels.
        Returns
        -------
        accuracy_score: float
            Accuracy of the model
        """
        predictions = self._predict(dataset)
        return accuracy(dataset.y, predictions)
