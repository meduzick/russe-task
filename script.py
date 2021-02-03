# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 19:02:00 2021

@author: User
"""
import tensorflow as tf
from input_pipeline import get_iterator, FileObj
from collections import namedtuple
from model import Model
from utils import eval_from_path, count_num_steps
import pandas as pd

dev_files = FileObj('./data/test_seq.p',
                      './data/test_queries.p',
                      './data/test_contexts.p',
                      './data/test_lens.p',
                      './data/test_target.p')

class HParams(namedtuple('hparams',
                         ['num_units',
                          'num_layers',
                          'num_classes',
                          'dropout_rate',
                          'learning_rate',
                          'batch_size',
                          'num_epochs',
                          'buffer_size',
                          'num_steps_to_eval',
                          'chkpts_dir'])):
  pass


hparams = HParams(num_units = 128,
                  num_layers = 3,
                  num_classes = 8,
                  dropout_rate = 0.3,
                  learning_rate = 3e-04,
                  batch_size = 64,
                  num_epochs = 20,
                  buffer_size = 512,
                  num_steps_to_eval = 10,
                  chkpts_dir = './model/chkpts')

test = pd.read_csv('./data/test.csv', sep = '\t')

def run():
    dev_graph = tf.Graph()
        
    with dev_graph.as_default():
        
        dev_iterator, dev_total_num = get_iterator(regime = 'DEV',
                                                  buffer_size = None,
                                                  num_epochs = 1,
                                                  batch_size = hparams.batch_size,
                                                   filesobj = dev_files)
    
        dev_model = Model(hparams,
                          dev_iterator,
                          'DEV')    
        
        tf.get_default_graph().finalize()
    
    dev_session = tf.Session(graph = dev_graph)
    
    eval_steps = count_num_steps(1,
                                dev_total_num,
                                hparams.batch_size)
    
    ari = eval_from_path('./best_shot/chkpts-130',
                   dev_model,
                   dev_session,
                   dev_iterator,
                   eval_steps,
                   test
                   )
    print('score on the dev set = {}'.format(ari))
    
if __name__ == '__main__':
    
    run()