# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 19:47:26 2021

@author: User
"""
from sklearn.cluster import KMeans
import numpy as np
from collections import namedtuple

class Config(namedtuple('model_config',
                        ['algorithm',
                         'column_scaling',
                         'normalization',
                         'kmeans'])):
    
    pass


def kmeans_clusterization(contexts,
                          column_scaling,
                          normalization,
                          n_clusters,
                          n_init,
                          max_iter,
                          algorithm,
                          random_state):
    
    norms = None
    mean = None
    std = None
    
    if column_scaling:
        norms = np.linalg.norm(contexts, ord = 2, axis = 0)
        norms = norms[np.newaxis, :]
        contexts = contexts / norms
    
    if normalization:
        mean = np.mean(contexts, axis = 0)[np.newaxis, :]
        std = np.std(contexts, axis = 0)[np.newaxis, :]
        contexts = (contexts - mean) / std
        
        
    kmeans = (KMeans(n_clusters = n_clusters,
                     n_init = n_init,
                     max_iter = max_iter,
                    algorithm = algorithm,
                    random_state = random_state)
              .fit(contexts))
    
    return kmeans, norms, mean, std
    
    
    

class Clusterizer(object):
    '''
    ===========================================================
    Realization of clustering algorithms in one spot. 
    
    Clusterization algorithms are as follows:
        1. KMeans
        2. Hierarchical clustering
        3. Affine propagation
        4. Exploit linear properties of word2vec embeddings
    
    ===========================================================
    
    '''
    
    def __init__(self, config):
        
        self._algorithm = config.algorithm
        self._column_scaling = config.column_scaling
        self._normalization = config.normalization
        self._kmeans_features = config.kmeans
        
    def fit(self, X, y = None):
        '''
        ===========================================================
        Execute clusterization depends on the type of clusterization 
        and hyperparameters.
        
        Args:
            X (np.ndarray): numpy matrix of shape (num_contexts, embedding_dim)
            y (None): optional
            
        Returns:
            pd DataFrame: initial frame, but with predicted labels
        
        ===========================================================
        '''
                        
        if self._algorithm == 'kmeans':
            self.model, self.norms, self.mean, self.std = \
                        kmeans_clusterization(X,
                                              self._column_scaling,
                                              self._normalization,
                                              **self._kmeans_features)
                        
    def predict(self, X, y = None):
        '''
        ===========================================================
        Applies fitted model.
        
        Args:
            X (np.ndarray): numpy matrix of shape (num_contexts, embedding_dim)
            y (None): optional
            
        Returns:
            list: predicted labels
        
        ===========================================================
        '''
        if not hasattr(self, 'model'):
            raise NotImplementedError('You should fit the model first')
            
        if self._column_scaling:
            X = X / self.norms
            
        if self._normalization:
            X = (X - self.mean) / self.std
            
        labels = self.model.predict(X)
        
        return labels
            
        
            
            