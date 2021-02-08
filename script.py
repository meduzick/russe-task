# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 19:02:00 2021

@author: User
"""
from collections import namedtuple
import tensorflow as tf
import pandas as pd
import model
import utils
import input_pipeline

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
    '''
    ===========================================================
    Embodies configuration properties for evaluation function
    Args:
        num_units (int): size of rnn hidden state
        num_layers (int): number of layers in multilayer rnn
        num_classes (int): num of distinct gold sences ids
        dropout_rate (float): drop probability
        learning_rate (float): learning rate
        batch_size (int): size of the one batch
        num_epochs (int): number of epochs to perform training
                        process
        buffer_size (int): number of elements to perform
                            shuffling
        num_steps_to_eval (int): num of steps to cover
                                wholel dev dataset
        chkpts_dir (str): path to current checkpoint
    ===========================================================
    '''
    pass
    
def run():
    '''
    ===========================================================
    Function performs evaluation of model on the whole dev
    dataset. Preprocessed and prepared data comes from files
    specified in config object.
    ===========================================================
    '''
    dev_files = input_pipeline.FileObj(
                        './data/test_seq.p',
                        './data/test_queries.p',
                        './data/test_contexts.p',
                        './data/test_lens.p',
                        './data/test_target.p')
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
    dev_graph = tf.Graph()
    with dev_graph.as_default():
        dev_iterator, dev_total_num = input_pipeline.get_iterator(
                                            regime = 'DEV',
                                            buffer_size = None,
                                            num_epochs = 1,
                                            batch_size = hparams.batch_size,
                                            filesobj = dev_files)
        dev_model = model.Model(hparams,
                                dev_iterator,
                                'DEV')
        tf.get_default_graph().finalize()
    dev_session = tf.Session(graph = dev_graph)
    eval_steps = utils.count_num_steps(1,
                                       dev_total_num,
                                       hparams.batch_size)
    ari = utils.eval_from_path('./best_shot/chkpts-130',
                               dev_model,
                               dev_session,
                               dev_iterator,
                               eval_steps,
                               test)
    print('score on the dev set = {}'.format(ari))
    
if __name__ == '__main__':
    run()
    