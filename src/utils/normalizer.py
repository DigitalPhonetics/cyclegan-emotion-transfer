import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

#=========================================================================
#
#  This file includes different data normalization methods.             
#
#=========================================================================

def normalize_cv(X, y, i, norm="zero_score"):
    """
    Normalize input data for cross validation
    @param X: List of sessions for feature data
    @param y: List of sessions for labels
    @param i: Session id for test set
    @param norm: Normalization type
    @return: Normalized training and test data where session i as test set
    and the rest sessions as training set
    """
    X_test = X[i]
    y_test = y[i]
    X_train = pd.concat(X[:i] + X[i+1:])
    y_train = pd.concat(y[:i] + y[i+1:])
    if norm == "min_max":
        scaler = preprocessing.MinMaxScaler()
    elif norm == "max_abs":
        scaler = preprocessing.MaxAbsScaler()
    else:
        scaler = preprocessing.StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), index=y_train.index.values)
    X_train.columns = X[i].columns.values
    X_test = pd.DataFrame(scaler.transform(X_test), index=y_test.index.values)
    X_test.columns = X[i].columns.values
    return X_train, X_test, y_train, y_test
    
    
def normalize(data, norm="zero_score"):
    """
    Normalize input data for the entire set
    @param data: Feature data to be normalized
    @param norm: Normalization type
    @return: Normalized feature data
    """
    if norm == "min_max":
        scaler = preprocessing.MinMaxScaler()
    elif norm == "max_abs":
        scaler = preprocessing.MaxAbsScaler()
    else:
        scaler = preprocessing.StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data), index=data.index.values)
    data_scaled.columns = data.columns.values
    return data_scaled
