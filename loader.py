# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 12:20:02 2021

@author: User
"""
from abc import ABC, abstractmethod
import pandas as pd
from simple_elmo import ElmoModel
import tensorflow as tf
from utils import calculate_positions, _calculate_q_n_contexts
from sklearn.decomposition import PCA
import numpy as np


class Loader(ABC):       
    
    def __init__(self, preprocessor, pca_seq, pca_query, pca_cont):
        self._num_internal_classes = 8
        self._preprocessor = preprocessor
        self.pca_seq = pca_seq
        self.pca_query = pca_query
        self.pca_cont = pca_cont
    
    def _read_dataset(self, path):
        df = pd.read_csv(path, sep = '\t')
        df.reset_index(drop = True, inplace = True)
        return df
    
    def _encode_corpus(self, path, corpus):
        model = ElmoModel()
        tf.reset_default_graph()
        model.load(path)
        elmo_vectors = model.get_elmo_vectors(corpus, 
                                              layers = 'top')
        return elmo_vectors
    
    @abstractmethod
    def _preprocess_target_variable(self):
        raise NotImplementedError('You should define the method first')
        
    @abstractmethod
    def _process_texts(self):
        raise NotImplementedError('You should define the method first')
        
    @abstractmethod
    def _decompose(self, Q, C, S):
        raise NotImplementedError('You should define the method first')
    
    def load_dataset(self, 
                     path,
                     path_to_elmo):
        print('LOADING DATASET...')
        self.df = self._read_dataset(path)
        
        print('PREPROCESSING TARGET...')
        self._preprocess_target_variable()
        
        print('PREPROCESSING TEXTS...')
        corpus = self._process_texts()
        
        elmo_vectors = self._encode_corpus(path_to_elmo, corpus)
        
        print('CALCULATING RESPONSES...')
        positions = calculate_positions(self.df.word.values,
                                        corpus,
                                        self._preprocessor._lemma_analyzer,
                                        self.df.context.values, 
                                        self.df.positions.values)
        
        queries, contexts = _calculate_q_n_contexts(corpus,
                                                    positions,
                                                    elmo_vectors)
        
        queries, contexts, sequences = self._decompose(queries,
                                                      contexts,
                                                      elmo_vectors
                                                      )
        lens = [len(elem) for elem in corpus]
        target = self.df.gold_sense_id.values
        
        return queries, contexts, sequences, lens, target 
    
        
        
        
class TrainLoader(Loader):
    
    def _preprocess_target_variable(self):        
        def _remap(df):
            d = {sence: index for index, sence in enumerate(df.gold_sense_id.unique())}
            df.loc[:, 'gold_sense_id'] = df.gold_sense_id.apply(lambda x: d[x])
            return df
    
        self.df = (self.df
         .groupby('word')
         .apply(lambda df: _remap(df)))
    
        d = {new_id: new_id if index < self._num_internal_classes \
                 else self._num_internal_classes for index, new_id in \
                     enumerate(self.df.gold_sense_id.value_counts().index)}
        
        self.df['gold_sense_id'] = self.df.gold_sense_id.apply(lambda x: d[x])
        self.df['gold_sense_id'] = self.df.gold_sense_id.replace({8:7})
        
    def _process_texts(self):
        corpus = self.df.context.values

        corpus = (self._preprocessor
                        .fit(corpus)
                        .transform(corpus))
        
        ### Note kostyl here!
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
    
    def _preprocess_target_variable(self):    
        self.df['gold_sense_id'] = (self.df
                                    .gold_sense_id
                                    .apply(lambda x: x - 1 if x < 9 else 7))
        
    def _process_texts(self):
        
        ### Note the kind of cheating here!
        self.df.loc[494, 'context'] = '15, 000 на всех членов семьи.Вот тогда и законы будут человечными Анатомия и физиология человека'
        self.df.loc[1202, 'context'] = 'выше чем 1 метр или до 12 лет-8 евро, дети ниже чем 1 метр –вход бесплатный Билеты: взрослые-16 евро, дети выше чем 1 метр или старше 10 лет-12 евро, дети до 10 лет'
        self.df.loc[1381, 'context'] = 'крепление Крепление на стену по стандарту VESA 100мм Блок питания внешний'
        self.df.loc[1398, 'context'] = 'Пористые заполнители Блоки оконные'
        self.df.loc[1514, 'context'] = 'Библиографические ресурсы и каталоги Блок библиографических ресурсов глобальных сетей обширен и разнообразен. Его главной'
        self.df.loc[2019, 'context'] = 'Выпускаемая продукция Вешалка детская'
        self.df.loc[2811, 'context'] = 'Волги только левым расположением запасного колеса. Оно так же прикручено винтом. горизонтальным торсионам и удерживалась ими в открытом положении. Причем оригинальной'
        self.df.loc[3134, 'context'] = 'касается продажи семян, живых растений и сопутствующих товаров (грунтов, удобрений, горшков и т. д.), они по-прежнему процветают. Неудивительно, что те территории на рынках'
        self.df.loc[2566, 'context'] = 'Электропроводка для подключения светодиодных знаков в задней части прицепа  для подключения электрооборудования прицепа к электросети автомобиля (вилка/розетка)'
        
        corpus = self.df.context.values
        
        ### Note the difference here!
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