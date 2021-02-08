# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 12:20:02 2021

@author: User
"""
from collections import namedtuple
import pickle as p
import tensorflow as tf

class DataPack(namedtuple('datapack',
                          ['sequences',
                           'queries',
                           'contexts',
                           'lens',
                           'target',
                           'total_num'])):
    '''
    ============================================================
    Used to store raw data before passing it to input generator.
    Allow to access data through attributes.
    Args:
        sequences (np.array): batch of elmo embeddings for contexts
        queries (np.array): batch of elmo embeddings for lemmas
        contexts (np.array): batch of elmo embeddings for
                            surrounding tokens
        lens (list): list of integers
        target (list): list of integers
        total_num (int): size of the input data
    ============================================================
    '''
    pass

class FileObj(namedtuple('files',
                         ['sequences',
                          'queries',
                          'contexts',
                          'lens',
                          'target'])):
    '''
    ============================================================
    Used to store filenames for training and evaluation.
    Args:
        sequences (str): path to sequences file
        queries (str): path to queries file
        contexts (str): path to contexts file
        lens (str): path to lens file
        target (str): path to target file
    ============================================================
    '''
    pass

class Batch(namedtuple('batch',
                       ['sequence',
                        'query',
                        'context',
                        'lens',
                        'target',
                        'initializer'])):
    '''
    ============================================================
    Used to store attributes of corresponding tensorflow generator
    Args:
        sequence (tf Tensor): 3d tensor with embedded sequences
        query (tf Tensor): 2d tensor with embedded lemmas
        context (tf Tensor): 2d tensor with embedded contexts
        lens (tf Tensor): 1d tensor with corresponding lengths
        target (tf Tensor): 1d tensor with targets
        initializer (tf Op): initialization op
    ============================================================
    '''
    pass

def _get_data(filesobj):
    sequences = p.load(open(filesobj.sequences, 'rb'))
    queries = p.load(open(filesobj.queries, 'rb'))
    contexts = p.load(open(filesobj.contexts, 'rb'))
    lens = p.load(open(filesobj.lens, 'rb'))
    target = p.load(open(filesobj.target, 'rb'))
    return DataPack(sequences = sequences,
                    queries = queries,
                    contexts = contexts,
                    lens = lens,
                    target = target,
                    total_num = len(target))

def _generator(sequences, queries, contexts, lens, targets):
    for sequence, query, context, length, target in\
        zip(sequences, queries, contexts, lens, targets):
            yield sequence, query, context, length, target

def _get_batched_input(generator,
                       regime,
                       buffer_size,
                       num_epochs,
                       batch_size):
    dataset = tf.data.Dataset.from_generator(generator,
                                            output_shapes = (tf.TensorShape([None,
                                                                             300]),
                                                             tf.TensorShape([300]),
                                                             tf.TensorShape([300]),
                                                             tf.TensorShape([]),
                                                             tf.TensorShape([])),
                                            output_types = (tf.float32,
                                                            tf.float32,
                                                            tf.float32,
                                                            tf.int32,
                                                            tf.int32))
    if regime == 'TRAIN':
        dataset = (dataset
                   .shuffle(buffer_size,
                            reshuffle_each_iteration = True)
                   .repeat(num_epochs))
    dataset = (dataset
               .batch(batch_size,
                      drop_remainder = False))
    iterator = dataset.make_initializable_iterator()
    sequence, query, context, lens, target = iterator.get_next()
    return Batch(sequence = sequence,
                 query = query,
                 context = context,
                 lens = lens,
                 target = target,
                 initializer = iterator.initializer)

def get_iterator(regime,
                 buffer_size,
                 num_epochs,
                 batch_size,
                 filesobj):
    '''
    ============================================================
    Used to get tensorflow generator with bathced tensors.
    Args:
        regime (str): mode
        buffer_size (int): num of elements to shuffle
        num_epochs (int): num of epochs to run training
        batch_size (int): size of the batch
        filesobj (FileObj): filenames
    ============================================================
    '''
    with tf.name_scope('inputs'):
        data = _get_data(filesobj)
        generator = lambda: _generator(data.sequences,
                                       data.queries,
                                       data.contexts,
                                       data.lens,
                                       data.target)
        iterator  = _get_batched_input(generator,
                                       regime,
                                       buffer_size,
                                       num_epochs,
                                       batch_size)
    return iterator, data.total_num
