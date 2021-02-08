# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 12:20:02 2021

@author: User
"""
import os
from os.path import join
from math import ceil
import numpy as np
from tqdm import tqdm_notebook as tqdm
import pandas as pd
from evaluate import evaluate

def _char_similarity(word, candidate):
    '''Counts number of unique subsequent symbols'''
    word = list(word)
    candidate = list(candidate)
    sim_score = 0
    for i in range(min(len(word), len(candidate))):
        if word[i] == candidate[i]:
            sim_score += 1
        else:
            break
    return sim_score

def _approximate_position(sentence, lexem):
    '''Selecting position with max similarity score'''
    scores = []
    for word in sentence:
        scores.append(_char_similarity(lexem, word))
    return np.argmax(scores)

def calculate_positions(lexemmes, corpus, analyzer):
    '''
    =============================================================
    Takes in lemmas and lemmatized corpus. Returns sentences,
    positions of the lemmas in the sentences and target.
    Args:
      lexemmes (list): list of lexemmes
      corpus (list): list of lists of strings
      analyzer (maru analyzer): instance of maru analyzer
      original_seqs (list): list of strings
      or_positions (list): list of strings
    Returns:
      positions
    =============================================================
    '''
    positions = []
    for lex, sentence in zip(lexemmes, corpus):
        lemma = list(analyzer.analyze([lex]))[0].lemma
        if lex in sentence:
            position = sentence.index(lex)
            positions.append(position)
        elif lemma in sentence:
            position = sentence.index(lemma)
            positions.append(position)
        else:
            position = _approximate_position(sentence, lex)
            positions.append(position)
    return positions

def _calculate_q_n_contexts(corpus,
                            positions,
                            elmo_vectors):
    queries = []
    contexts = []
    window_size = 5
    lens = [len(elem) for elem in corpus]
    for position, sent_matrix, seq_len in zip(positions, elmo_vectors, lens):
        row = sent_matrix[position, :]
        row = row / np.linalg.norm(row)
        queries.append(row)
        left_id = max(0, position - window_size)
        right_id = min(seq_len - 1, position + window_size)
        context = np.sum(sent_matrix[left_id:right_id + 1], axis = 0)
        context = context / np.linalg.norm(context)
        contexts.append(context)
    queries = np.array(queries)
    contexts = np.array(contexts)
    return queries, contexts

def copy_files(path):
    '''
    ============================================================================
    Copies checkpoint files that correspond to the best achieved eval metric
    in subdirectory.
    Args:
      path (str): path to best checkpoint
    ===========================================================================
    '''
    for elem in os.listdir('./model'):
        src = './model/' + elem
        dest = './model/best_shot/' + elem
        if src.find(path) >= 0:
            os.system('cp {} {}'.format(src, dest))

def clear_dir(path):
    '''
    ============================================================================
    Deletes all the files from the directory, where best checkpoint files are
    stored (previously uploaded checkpoint files).
    Args:
      path (str): subdirectory
    ===========================================================================
    '''
    for file in os.listdir(path):
        file = join(path, file)
        os.remove(file)

def eval_from_path(chkpt_path,
                   dev_model,
                   dev_sess,
                   dev_iterator,
                   num_dev_steps,
                   test):
    '''
    ============================================================================
    Evaluate model restored from the given path.
    Args:
      chkpt_path (str): path to model checkpoint
      dev_model (model.Model): needed to construct graph
      dev_sess (tf Session): session associated with constructed graph
      dev_iterator (BatchedInput): batch of tf Tensors from generator
      num_dev_steps (int): number of steps needed to cover whole dev set
      test (pd DataFrame): original dev dataset; needed to calculate ari
    Returns:
        ari score on the whole dev dataset
    ===========================================================================
    '''
    predictions = []
    dev_model.saver.restore(dev_sess,
                            chkpt_path)
    dev_sess.run(dev_iterator.initializer)
    with tqdm(total = num_dev_steps) as prog:
        for _ in range(num_dev_steps):
            res = dev_model.evaluate(dev_sess)
            predictions += res.predictions.tolist()
            prog.update(1)
    test['predict_sense_id'] = pd.Series(predictions,
                                         index = test.index)
    test.to_csv('solution.csv', sep = '\t', index = False)
    ari = evaluate('solution.csv')
    return ari

def deval(train_model,
          train_sess,
          chkpts_dir,
          current_step,
          dev_model,
          dev_sess,
          dev_iterator,
          num_dev_steps,
          test):
    '''
    ==========================================================================
    The function evaluate model from current checkpoint on the whole
     dev dataset.
    Args:
      train_model (object): instance of Model class
      train_sess (tf Session): instance of tf Session
      chkpts_dir (str): path to the checkpoints
      current_step (int): step
      dev_model (object): instance of Model class
      dev_sess (tf Session): instance of tf Session
      dev_iterator (generator):input data subgrah
      num_dev_steps (int): number of steps to process whole dataset (dev)
    ===========================================================================
    '''
    predictions = []
    current_chkpt_path = train_model.saver.save(train_sess,
                                                chkpts_dir,
                                                current_step)
    dev_model.saver.restore(dev_sess,
                            current_chkpt_path)
    dev_sess.run(dev_iterator.initializer)
    for _ in range(num_dev_steps):
        res = dev_model.evaluate(dev_sess)
        predictions += res.predictions.tolist()
    test['predict_sense_id'] = pd.Series(predictions,
                                         index = test.index)
    test.to_csv('solution.csv', sep = '\t', index = False)
    ari = evaluate('solution.csv')
    return ari, current_chkpt_path

def count_num_steps(num_epochs,
                    data_size,
                    batch_size,
                    include_last_batch = True):
    '''
    ============================================================================
    Calculates number of steps needed to process whole dataset num_epochs times
    with the batch of size batch_size.
    Args:
      num_epochs (int): number of epochs
      data_size (int): number of samples in the dataset
      batch_size (int): batch size
      include_last_batch (bool): whether to include last batch or not
    Returns:
      (int): number of steps
    ===========================================================================
    '''
    if include_last_batch:
        return ceil(data_size * num_epochs / batch_size)
    return data_size * num_epochs // batch_size
  