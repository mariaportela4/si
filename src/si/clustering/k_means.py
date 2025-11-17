import numpy as np
from si.base.model import Model
from si.base.transformer import Transformer
from si.data import dataset
from si.data.dataset import Dataset
from si.statistics import euclidean_distance

class KMeans(Transformer, Model):
    def __init__(self,k=3,max_iter=100,distance= euclidean_distance,**kwargs):
        self.distance = distance

        self.centroids = None
        self.labels = None

    def _init_centroids (self, dataset):
        random_indexes = np.random.permutation(dataset.shape()[0])[:self.k]
        self.centroids= dataset.X[random_indexes, :]
    
    def _calculate_distances (self, sample):
        return self.distance(sample, self.centroids)
    
    def _get_closest_centroids (self, sample):
        centroids_distance = self._calculate_distances(sample)
        centroids_index = np.argmin(centroids_distance, axis=0) # retorna o index do valor minimo --> centroide mais proximo
        return centroids_index
    
    def _fit(self, dataset):
        self._init_centroids (dataset)

        i = 0
        convergence = False

        labels = np.zeros(dataset.shape()[0])
        while not convergence and i < self.max_inter:

            new_labels = np.apply_along_axis(self._get_closest_centroids, arr=dataset.X, axis=1)
            self.labels = new_labels

            centroids = []
            for j in range(self.k):
                mask = new_labels == j
                new_centroid = np.mean(dataset.X[mask])
                centroids.append(new_centroid)

            convergence = not np.any(new_labels != labels)
            labels = new_labels
            i += 1
        self.labels = labels
        return self

    def _transform (self, dataset):
        euclidean_distance = np.apply_along_axis(self._calculate_distances, arr=dataset.X, axis=1)
        return euclidean_distance
    
    def _predict (self, datatset):
        new_labels = np.apply_along_axis(self._get_closest_centroids, arr=dataset.X, axis=1)
        return new_labels