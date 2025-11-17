import numpy as np


def euclidean_distance (x: np.ndarray,y:np.ndarray) -> np.ndarray:
    return np.sqrt((x - y)**2).sum(axis=1) # axis = 1 --> soma das linhas ao longo do eixo 1