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
  
  pass

class FileObj(namedtuple('files',
                         ['sequences',
                          'queries',
                          'contexts',
                          'lens',
                          'target'])):
  
  pass

class Batch(namedtuple('batch',
                       ['sequence',
                        'query',
                        'context',
                        'lens', 
                        'target',
                        'initializer'])):
  
  pass


def _get_data(regime,
              filesobj):

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

def _generator(S, Q, C, lens, target):
  for s, q, c, l, t in zip(S, Q, C, lens, target):
    yield s, q, c, l, t

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
               .repeat(num_epochs)
                )
    
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
  

  with tf.name_scope('inputs'):
    
    data = _get_data(regime,
                     filesobj)
    
    iterator  = _get_batched_input(lambda: _generator(data.sequences,
                                                      data.queries,
                                                      data.contexts,
                                                      data.lens,
                                                      data.target),
                                    regime,
                                    buffer_size,
                                    num_epochs,
                                    batch_size
                                  )
    
  return iterator, data.total_num
