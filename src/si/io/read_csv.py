import pandas as pd
from si.data.dataset import Dataset
import numpy as np

def read_csv(filename: str,sep:str,features: bool,label: bool) -> Dataset:

    df = pd.read_csv(filename, sep=sep)

    if features and label:
        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[-1].to_numpy()
        feature_names= df.columns[:-1]
        label_names = df.columns[-1]
        return Dataset(X=X, y=y, features=feature_names, label=label_names)
    elif features:
        X = df.to_numpy
        features_names = df.columns
        return Dataset(X=X, features = features_names)
    elif label:
        X = np.array()
        y = df.iloc[:, -1].to_numpy
        return Dataset (X=X, y=y, label =label_names)
    else: 
        return None

def write_csv (filename: str, dataset:Dataset,sep:str,features: bool = False,label: bool = False) -> Dataset:
   
    df = pd.DataFrame(dataset.X, columns = dataset.features)

    if features:
        df.columns = dataset.features
    
    if label:
        y = dataset.y
        label_name = dataset.label
        df[label_name] = y
    
    else:
        y = None
        label_name = None
    
    df.to_csv(filename, sep=sep, index=False)