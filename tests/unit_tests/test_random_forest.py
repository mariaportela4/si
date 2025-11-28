import os
from unittest import TestCase

from datasets import DATASETS_PATH
from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.models.random_forest_classifier import RandomForestClassifier


class TestRandomForest(TestCase):
    def setUp(self):
        self.csv_file= os.path.join(DATASETS_PATH, 'iris','iris.csv')
        self.dataset= read_csv(filename=self.csv_file,features=True, label=True)
        self.train_dataset, self.test_dataset = train_test_split(self.dataset)
    
    def test_fit(self):
        n_trees = 10
        random_forest = RandomForestClassifier(n_estimators=n_trees)
        random_forest.fit(self.train_dataset)

        self.assertEqual(len(random_forest.trees),n_trees)

        for _, tree in random_forest.trees:
            self.assertEqual(tree.min_sample_split, 2)
            self.assertEqual(tree.max_depth, 10)
        
    def test_predict(self):
        random_forest = RandomForestClassifier(n_trees=10)
        random_forest.fit(self.train_dataset)
        predictions = random_forest.predict(self.test_dataset)

        self.assertEqual(predictions.shape[0], self.test_dataset.X.shape[0])
    
    def test_score(self):
        random_forest = RandomForestClassifier(n_trees=10)
        random_forest.fit(self.train_dataset)
        accuracy = random_forest.score(self.test_dataset)

        self.assertIsInstance(accuracy,float)
        self.assertGreaterEqual(accuracy,0)
        self.assertLessEqual(accuracy,1)
    
    def test_comparison_with_decision_tree(self):
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(self.train_dataset)
        decision_tree_accuracy = decision_tree.score(self.test_dataset)

        random_forest=RandomForestClassifier(n_trees=10, seed=45)
        random_forest.fit(self.train_dataset)
        random_forest_accuracy = random_forest.score(self.test_dataset)
    
        self.assertGreaterEqual(random_forest_accuracy,decision_tree_accuracy - 0.05)

        
    