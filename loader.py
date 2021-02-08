# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 12:20:02 2021

@author: User
"""
from abc import ABC, abstractmethod
import pandas as pd
from simple_elmo import ElmoModel
import tensorflow as tf
import numpy as np
import utils


class Loader(ABC):
    '''
    =================================================================
    Implements preprocessing pipeline:
        1. Apply preprocessor fit-transform
        2. Find target word in context and get embedding
        3. Get embeddings for the whole sequences
        4. Extract fixed-length context around target word and 
            embedd it
        5. Apply pca to reduce dimensionality
    =================================================================
    '''
    def __init__(self, preprocessor, pca_seq, pca_query, pca_cont):
        self._num_internal_classes = 8
        self._preprocessor = preprocessor
        self.pca_seq = pca_seq
        self.pca_query = pca_query
        self.pca_cont = pca_cont
        
    @abstractmethod
    def _preprocess_target_variable(self, frame):
        raise NotImplementedError('You should define the method first')
        
    @abstractmethod
    def _process_texts(self, frame):
        raise NotImplementedError('You should define the method first')
        
    @abstractmethod
    def _decompose(self, queries, contexts, sequences):
        raise NotImplementedError('You should define the method first')
        
    def load_dataset(self,
                     path,
                     path_to_elmo):
        '''
        =================================================================
        Implements pipeline steps.
        Args:
            path (str): path to source dataset
            path_to_elmo (str): path to pretrained elmo model
        Returns:
            queries (np.array): embeddings for all the queries
            contexts (np.array): embeddings for all the contexts
            sequences (np.array): embedded sequences
            lens (list): array of integers
            target (list): array of integers
        =================================================================
        '''
        def _read_dataset(path):
            frame = pd.read_csv(path, sep = '\t')
            frame.reset_index(drop = True, inplace = True)
            return frame
        
        def _encode_corpus(path, corpus):
            model = ElmoModel()
            tf.reset_default_graph()
            model.load(path)
            elmo_vectors = model.get_elmo_vectors(corpus,
                                                  layers = 'top')
            return elmo_vectors
        
        print('LOADING DATASET...')
        frame = _read_dataset(path)
        print('PREPROCESSING TARGET...')
        frame = self._preprocess_target_variable(frame)
        print('PREPROCESSING TEXTS...')
        corpus = self._process_texts(frame)
        elmo_vectors = _encode_corpus(path_to_elmo, corpus)
        print('CALCULATING RESPONSES...')
        positions = utils.calculate_positions(frame.word.values,
                                              corpus,
                                              self._preprocessor.lemma_analyzer)
        queries, contexts = utils._calculate_q_n_contexts(corpus,
                                                          positions,
                                                          elmo_vectors)
        queries, contexts, sequences = self._decompose(queries,
                                                       contexts,
                                                       elmo_vectors)
        lens = [len(elem) for elem in corpus]
        target = frame.gold_sense_id.values
        return queries, contexts, sequences, lens, target
    
class TrainLoader(Loader):
    '''
    =================================================================
    Inherits from Loader and implements input pipeline for
    training process:
        1. Apply preprocessor fit-transform
        2. Find target word in context and get embedding
        3. Get embeddings for the whole sequences
        4. Extract fixed-length context around target word and
            embedd it
        5. Fit pca instances
    =================================================================
    '''
    def _preprocess_target_variable(self, frame):
        
        def _remap(sub_frame):
            id2index = {sence: index for index, sence in enumerate((sub_frame
                                                                    .gold_sense_id
                                                                    .unique()))}
            sub_frame.loc[:, 'gold_sense_id'] = (sub_frame
                                                 .gold_sense_id
                                                 .apply(lambda x: id2index[x]))
            return sub_frame
        
        frame = (frame
                 .groupby('word')
                 .apply(_remap))
        index2target = {new_id: new_id \
                        if index < self._num_internal_classes \
                        else self._num_internal_classes for index, new_id in \
                        enumerate(frame.gold_sense_id.value_counts().index)}
        frame['gold_sense_id'] = frame.gold_sense_id.apply(lambda x: index2target[x])
        frame['gold_sense_id'] = frame.gold_sense_id.replace({8:7})
        return frame
    
    def _process_texts(self, frame):
        corpus = frame.context.values
        corpus = (self._preprocessor
                      .fit(corpus)
                      .transform(corpus))
        corpus[1310][-1] = 'жила'
        corpus[1311][0] = 'жила'
        corpus[1312][-2] = 'жила'
        corpus[1313][5] = 'жила'
        corpus[1318][1] = 'жила'
        corpus[304][8] = 'дело'
        return corpus
    
    def _decompose(self,
                   queries,
                   contexts,
                   sequences):
        sequences = np.reshape(self.pca_seq.fit_transform(np.reshape(sequences, (-1, 1024))),
                               (len(sequences), -1, 300))
        queries = self.pca_query.fit_transform(queries)
        contexts = self.pca_cont.fit_transform(contexts)
        return queries, contexts, sequences

class DevLoader(Loader):
    '''
    =================================================================
    Inherits from Loader and implements input pipeline for
    validation process:
        1. Apply preprocessor transform
        2. Find target word in context and get embedding
        3. Get embeddings for the whole sequences
        4. Extract fixed-length context around target word and
            embedd it
        5. Apply pca transform
    =================================================================
    '''
    def _preprocess_target_variable(self, frame):
        frame['gold_sense_id'] = (frame
                                  .gold_sense_id
                                  .apply(lambda x: x - 1 if x < 9 else 7))
        return frame
        
    def _process_texts(self, frame):
        frame.loc[494, 'context'] = '''15, 000 на всех членов семьи.Вот тогда
                    и законы будут человечными Анатомия и физиология человека'''
        frame.loc[1202, 'context'] = '''выше чем 1 метр или до 12 лет-8 евро,
                дети ниже чем 1 метр –вход бесплатный Билеты: взрослые-16 евро,
                дети выше чем 1 метр или старше 10 лет-12 евро, дети до 10 лет'''
        frame.loc[1381, 'context'] = '''крепление Крепление на стену по
                                    стандарту VESA 100мм Блок питания внешний'''
        frame.loc[1398, 'context'] = '''Пористые заполнители Блоки оконные'''
        frame.loc[1514, 'context'] = '''Библиографические ресурсы и каталоги
                                    Блок библиографических ресурсов глобальных 
                                    сетей обширен и разнообразен. Его главной'''
        frame.loc[2019, 'context'] = '''Выпускаемая продукция Вешалка детская'''
        frame.loc[2811, 'context'] = '''Волги только левым расположением
                                запасного колеса. Оно так же прикручено винтом. 
                                горизонтальным торсионам и удерживалась ими в
                                открытом положении. Причем оригинальной'''
        frame.loc[3134, 'context'] = '''касается продажи семян, живых
                        растений и сопутствующих товаров (грунтов, удобрений, 
                        горшков и т. д.), они по-прежнему процветают.
                        Неудивительно, что те территории на рынках'''
        frame.loc[2566, 'context'] = '''Электропроводка для подключения
                светодиодных знаков в задней части прицепа  для подключения 
                электрооборудования прицепа к электросети автомобиля (вилка/розетка)'''
        corpus = frame.context.values
        corpus = (self._preprocessor
                      .transform(corpus))
        return corpus
    
    def _decompose(self,
                   queries,
                   contexts,
                   sequences):
        sequences = np.reshape(self.pca_seq.transform(np.reshape(sequences, (-1, 1024))),
                               (len(sequences), -1, 300))
        queries = self.pca_query.transform(queries)
        contexts = self.pca_cont.transform(contexts)
        return queries, contexts, sequences
    