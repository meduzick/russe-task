# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 14:16:24 2021

@author: User
"""
from collections import namedtuple
from abc import ABC, abstractmethod
import numpy as np

class Config(namedtuple('vectorizer_config',
                        ['embedding_model_type',
                         'model',
                         'average_type',
                         'min_pooling',
                         'max_pooling',
                         'concatenation',
                         'return_type'])):
    '''
    ===========================================================
    Embodies configuration properties for the vectorizer.
    Args:
        embedding_model_type (str): 'gensim', other will be
                                added if necessary, examples
                                from failed experiments are
                                removed
        model (class instance): pretrained model itself
        average_type (str): simple average or weighted average
        min_pooling (bool): whether to perform min pooling
                            on embedding tensor
        max_pooling (bool): whether to perform max pooling
                            on embedding tensor
        concatenation (str): whether to concatenate average
                            with pooling or not
        return_type (str): whether to return concatenated
                            pooling, particular pooling or
                            average concatenated with pooling
    Returns:
        list of numpy arrays of shape (seq_len, embedding_dim)
    ===========================================================
    '''
    pass

class Layer(ABC):
    '''
    ===========================================================
    Implements the general layer functionality.
    There is only one functional option, but more is to be added
    soon if needed.
    ===========================================================
    '''
    @abstractmethod
    def __call__(self, X):
        raise NotImplementedError('you should implement the method first')
        
class Embedding(Layer):
    '''Embeds token sequences with appropriate embedding vectors.'''
    def __init__(self, model_type, model):
        self._model_type = model_type
        self._model = model
        
    def __call__(self, X):
        '''
        ===========================================================
        Maps sequence of tokens to sequence of vectors.
        Args:
            X (list): list of lists of strings
        Returns:
            list of numpy arrays of shape (seq_len, embedding_dim)
        ===========================================================
        '''
        res = []
        if self._model_type == 'gensim':
            sub = np.random.normal(size = (self._model.vector_size))
            for line in X:
                line_embedded = []
                for word in line:
                    if word in self._model:
                        line_embedded.append(self._model[word])
                    else:
                        continue
                if len(line_embedded) == 0:
                    line_embedded.append(sub)
                res.append(line_embedded)
        return res
    
class Normalizer(Layer):
    '''Perform dimensionality reduction with possible concatenation.'''
    def __init__(self, average_type,
                 min_pooling,
                 max_pooling,
                 concat,
                 return_type):
        self._average_type = average_type
        self._min_pooling = min_pooling
        self._max_pooling = max_pooling
        self._concatenation = concat
        self._return_type = return_type
        
    def __call__(self, X):
        '''
        ===========================================================
        Reduce embeddings tensors to vectors.
        Args:
            X (list): list of numpy ndarrays
        Returns:
            numpy matrix of shape (num_context, embedding_dim)
        ===========================================================
        '''
        total_res = []
        pooling_res = []
        if self._average_type == 'simple_average':
            average_res = []
            for tensor in X:
                tensor = np.array(tensor)
                average_res.append(np.mean(tensor, axis = 0))
            average_res = np.array(average_res)
            total_res.append(average_res)
        if self._min_pooling:
            min_pooling_res = []
            for tensor in X:
                tensor = np.array(tensor)
                min_pooling_res.append(np.min(tensor, axis = 0))
            min_pooling_res = np.array(min_pooling_res)
            total_res.append(min_pooling_res)
            pooling_res.append(min_pooling_res)
        if self._max_pooling:
            max_pooling_res = []
            for tensor in X:
                tensor = np.array(tensor)
                max_pooling_res.append(np.max(tensor, axis = 0))
            max_pooling_res = np.array(max_pooling_res)
            total_res.append(max_pooling_res)
            pooling_res.append(max_pooling_res)
        if self._concatenation == 'full':
            res = np.concatenate(total_res, axis = 1)
        if self._concatenation == 'pooling':
            res = np.concatenate(pooling_res, axis = 1)
        if self._return_type == 'only_average':
            return average_res
        if self._return_type == 'only_min_pooling':
            return min_pooling_res
        if self._return_type == 'only_max_pooling':
            return max_pooling_res
        return res

class Vectorizer():
    '''
    ===========================================================
    Vectorizes contexts: maps list of tokens (str) to fixed
    dimension real vector.
    Vectorization can be done:
        1. Embedding layer (pretrained embeddings)
        2. Averaging along 0 axis
        2.1 Weighted average along 0 axis
        2.2 Mean, max, min pooling
        2.3 Arbitrary concatenation
    ===========================================================
    '''
    def __init__(self, config):
        self._embedding_model_type = config.embedding_model_type
        self._embedding_model = config.model
        self._average_type = config.average_type
        self._min_pooling = config.min_pooling
        self._max_pooling = config.max_pooling
        self._concatenation = config.concatenation
        self._return_type = config.return_type
        
    def fit(self, X, y = None):
        self.embedding_layer = Embedding(self._embedding_model_type,
                                         self._embedding_model)
        self.normalizer = Normalizer(self._average_type,
                                     self._min_pooling,
                                     self._max_pooling,
                                     self._concatenation,
                                     self._return_type)
        return self
    
    def transform(self, X, y = None):
        '''
        ===========================================================
        Execute vectorization steps.
        Args:
            X (list): list of lists of strings
            y (None): ----
        Returns:
            np.ndarray of shape (num_context, embeddings_dim)
        ===========================================================
        '''
        if not hasattr(self, 'embedding_layer'):
            raise NotImplementedError('fit the vectorizer first')
        if not hasattr(self, 'normalizer'):
            raise NotImplementedError('fit the vectorizer first')
        x_embedded = self.embedding_layer(X)
        x_embedded_norm = self.normalizer(x_embedded)
        return x_embedded_norm
    