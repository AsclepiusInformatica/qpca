# -*- coding: utf-8 -*-
"""
Created on Thu Mar 7 14:33:34 2019

@author: finn
"""

from sklearn.decomposition import PCA
import numpy as np

def qPCA(M, C):
    '''qPCA function for mixture decomposition.
    ----------
    Parameters

    M: responses vector of mixture.
    C: pure components data array in shape of n_components x n_features.
    ----------
    Returns

    K: n_samples x n_components array, represents ratio of each component within each mixture.
    '''
    n_components = C.shape[0]
    pca = PCA(n_components=n_components-1)
    X = np.vstack((C, M))
    scores = pca.fit_transform(X)
    K = np.linalg.solve(np.vstack((scores[:-1].T, np.ones((1, n_components)))), np.vstack((scores[-1], 1)))
    return K

def exclude_trim(K, max_dev=1):
    '''Exclude background and trim spillovers for qPCA results.
    ----------
    Parameters
    
    K: ratio array calculated by qPCA.
    max_dev: non-negative float, max deviation allowed for spillovers, default to 1.
             ratios deviate larger than this value from normal boundary 0 and 1 will be treated as background and excluded from imaging results,
             while those deviate smaller than this value will be treated as spillovers and trimmed to 0 or 1.
    ----------
    Returns

    Knew: new ratio array after exclusion and spillovers trimming.
    '''
    Knew = K.copy()

    Knew[np.logical_or(Knew>max_dev+1, Knew<-max_dev)] = np.nan

    Knew[Knew<0] = 0
    Knew[Knew>1] = 1
    return Knew
