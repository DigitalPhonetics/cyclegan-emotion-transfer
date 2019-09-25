import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sin,cos,sqrt

#=========================================================================
#
#  This file defines helper functions for sample generation.             
#
#=========================================================================
                
def onehot(labels, n=4):
    """
    Convert emotion labels to onehot vector
    @param labels: List of emotion labels, e.g. [0, 1, 3, 0, 2]
    @param n: Number of emotion types
    @return: Onehot vector of emotion labels
    """
    res = []
    for label in labels:
        v = [0.]*n
        v[label] = 1.0
        res.append(v)
    return np.array(res)


def gaussian_mixture(y):
    """
    Generate 2d points following the distribution of a 4-component-GMM, 
    each emotion corresponds to a component
    @param y: List of emotion labels, e.g. [0, 1, 3, 0, 2]
    @return: Onehot vector of emotion labels
    """
    mean = np.array([[10, 0],
                     [0, -10], 
                     [-10, 0], 
                     [0, 10]
                    ])
    cov = np.array([[[10, 0], [0, 1]],
                    [[1, 0], [0, 10]],
                    [[10, 0], [0, 1]],
                    [[1, 0], [0, 10]]
                   ])
    code_vectors = [np.random.multivariate_normal(mean[i], cov[i]) for i in y]
    return np.array(code_vectors)
