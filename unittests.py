# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 12:20:23 2021

@author: User
"""
from preprocessor import Preprocessor, Config
from vectorizer import Vectorizer, Config as VConfig
import maru
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords_set = set(stopwords.words('russian'))
import collections
import razdel
from gensim.models import KeyedVectors
import pickle
import numpy as np


compare = lambda x, y: collections.Counter(x) == collections.Counter(y)

def run_preprocessor_unittests():
    
    def test1():
        '''
        ===========================================================
        Full house:
            1. Leave only alphabet characters
            2. Remove stop words
            3. Lemmatize and add pos tags
            
        ===========================================================
        
        '''
        
        print('running preprocessor test 1 ...')
    
        pattern = re.compile(r'[^а-яА-я ё]')
        analyzer = maru.get_analyzer(tagger='linear')
        config = Config(regexp = pattern,
                         stopwords = stopwords_set, 
                         analyzer = analyzer,
                         with_pos_tag = True, 
                         remove_stop_words = True,
                         lemmatize = True,
                         tokenizer = None)
        
        preprocessor = Preprocessor(config)
        
        case = ['Так говорила в июле 1805 года известная',
                '— Как можно быть здоровой... когда нравственно страдаешь?',
                'праздник отменен, Je vous avoue que toutes ces fêtes']
        
        expected = [['говорить_VERB', 'июль_NOUN', 'год_NOUN', 'известный_ADJ'],
                    ['здоровый_ADJ', 'нравственно_ADV', 'страдать_VERB'],
                    ### here is the case when lemmatization fails
                    ['праздник_NOUN', 'отменный_ADJ']] 
        
        res = preprocessor.fit(case).transform(case)
        
        for x, y in zip(res, expected):
            
            assert compare(x, y), 'failed with {} and {}'.format(x, y)
            
        print('test 1 passed')
        
    def test2():
        '''
        ===========================================================
        Following steps:
            1. Add numbers and punct
            2. No tokenizer
            3. Stop words
            4. pos tag
            
        ===========================================================
        
        '''
        
        print('running preprocessor test 2 ...')
    
        pattern = re.compile(r'[^а-яА-я0-9,.!?;\- ё]')
        analyzer = maru.get_analyzer(tagger='linear')
        config = Config(regexp = pattern,
                         stopwords = stopwords_set, 
                         analyzer = analyzer,
                         with_pos_tag = True, 
                         remove_stop_words = False,
                         lemmatize = True,
                         tokenizer = None)
        
        preprocessor = Preprocessor(config)
        
        case = ['Так говорила в июле 1805 года известная',
                '— Как можно быть здоровой... когда нравственно страдаешь?',
                'праздник отменен, Je vous avoue que toutes ces fêtes']
        
        expected = [['так_ADV', 'говорить_VERB', 'в_ADP', 'июль_NOUN',
                     '1805_NUM', 'год_NOUN', 'известный_ADJ'],
                    ['как_CONJ', 'можно_ADJ', 'быть_VERB', 'здоровой..._CONJ',
                     'когда_CONJ','нравственно_ADV', 'страдаешь?_PRON'],
                    ['праздник_NOUN', 'отменен,_VERB']] 
        
        res = preprocessor.fit(case).transform(case)
        
        for x, y in zip(res, expected):
            
            assert compare(x, y), 'failed with {} and {}'.format(x, y)
            
        print('test 2 passed')
        
    def test3():
        '''
        ===========================================================
        Following steps:
            1. Add numbers and punct
            2. Razdel tokenizer
            3. Stop words
            4. No pos tag
            5. Lemmatize
            
        ===========================================================
        
        '''
        
        print('running preprocessor test 3 ...')
    
        pattern = re.compile(r'[^а-яА-я0-9,.!?;\- ё]')
        analyzer = maru.get_analyzer(tagger='linear')
        tokenizer = razdel
        config = Config(regexp = pattern,
                         stopwords = stopwords_set, 
                         analyzer = analyzer,
                         with_pos_tag = False, 
                         remove_stop_words = False,
                         lemmatize = True,
                         tokenizer = tokenizer)
        
        preprocessor = Preprocessor(config)
        
        case = ['Так говорила в июле 1805 года известная',
                '— Как можно быть здоровой... когда нравственно страдаешь?',
                'праздник отменен, Je vous avoue que toutes ces fêtes']
        
        expected = [['так', 'говорить', 'в', 'июль',
                     '1805', 'год', 'известный'],
                    ['как', 'можно', 'быть', 'здоровый', '...', 'когда',
                     'нравственно', 'страдать', '?'],
                    ['праздник', 'отменный', ',']] 
        
        res = preprocessor.fit(case).transform(case)
        
        for x, y in zip(res, expected):
            
            assert compare(x, y), 'failed with {} and {}'.format(x, y)
            
        print('test 3 passed')
        
    def test4():
        '''
        ===========================================================
        Following steps:
            1. Add numbers and punct
            2. Razdel tokenizer
            3. Stop words
            4. No pos tag
            5. No lemmatization
            
        ===========================================================
        
        '''
        
        print('running preprocessor test 3 ...')
    
        pattern = re.compile(r'[^а-яА-я0-9,.!?;\- ё]')
        analyzer = maru.get_analyzer(tagger='linear')
        tokenizer = razdel
        config = Config(regexp = pattern,
                         stopwords = stopwords_set, 
                         analyzer = analyzer,
                         with_pos_tag = False, 
                         remove_stop_words = False,
                         lemmatize = False,
                         tokenizer = tokenizer)
        
        preprocessor = Preprocessor(config)
        
        case = ['Так говорила в июле 1805 года известная',
                '— Как можно быть здоровой... когда нравственно страдаешь?',
                'праздник отменен, Je vous avoue que toutes ces fêtes']
        
        expected = [['так', 'говорила', 'в', 'июле', '1805', 'года', 'известная'],
                    ['как', 'можно', 'быть', 'здоровой', '...', 'когда',
                     'нравственно', 'страдаешь', '?'],
                    ['праздник', 'отменен', ',']] 
        
        res = preprocessor.fit(case).transform(case)
        
        for x, y in zip(res, expected):
            
            assert compare(x, y), 'failed with {} and {}'.format(x, y)
            
        print('test 4 passed')
        
        
    test1()
    test2()
    test3()
    test4()
    
    
def run_vectorizer_unittests():
    
    def test1():
        '''
        ===========================================================
        Full house:
            1. Simple average
            2. Min pooling
            3. Max pooling
            4. Full concatenation
            5. Full return
            
        ===========================================================
        
        '''
        
        print('running vectorizer test 1 ...')
        
        MODEL_FILE = 'model.bin'
        model = KeyedVectors.load_word2vec_format(MODEL_FILE, binary=True)
        
        config = VConfig('gensim',
                         model = model,
                         average_type = 'simple_average',
                         min_pooling = True,
                         max_pooling = True,
                         concatenation = 'full',
                         return_type = 'full')
        
        vectorizer = Vectorizer(config)
        
        case = [['отвергнуть_VERB', 'щедрый_ADJ', 'дар_NOUN'],
            ['покупать_VERB', 'преданность_NOUN', 'дар_NOUN', 'награда_NOUN'],
            ['яд_NOUN', 'последний_ADJ', 'дар_NOUN', 'мой_DET', 'изора_NOUN']]
        
        expected = pickle.load(open('./sanity_checks_data/vectorizer1.p', 'rb'))
        
        res = vectorizer.fit(case).transform(case)
        
        np.testing.assert_allclose(expected, res)
        
        print('vectorizer test 1 is passed')
        
    def test2():
        '''
        ===========================================================
        Configuration:
            1. Simple average
            5. Full average
            
        ===========================================================
        
        '''
        
        print('running vectorizer test 2 ...')
        
        MODEL_FILE = 'model.bin'
        model = KeyedVectors.load_word2vec_format(MODEL_FILE, binary=True)
        
        config = VConfig('gensim',
                         model = model,
                         average_type = 'simple_average',
                         min_pooling = False,
                         max_pooling = False,
                         concatenation = None,
                         return_type = 'only_average')
        
        vectorizer = Vectorizer(config)
        
        case = [['отвергнуть_VERB', 'щедрый_ADJ', 'дар_NOUN'],
            ['покупать_VERB', 'преданность_NOUN', 'дар_NOUN', 'награда_NOUN'],
            ['яд_NOUN', 'последний_ADJ', 'дар_NOUN', 'мой_DET', 'изора_NOUN']]
        
        expected = pickle.load(open('./sanity_checks_data/vectorizer2.p', 'rb'))
        
        res = vectorizer.fit(case).transform(case)
        
        np.testing.assert_allclose(expected, res)
        
        print('vectorizer test 2 is passed')
        
    def test3():
        '''
        ===========================================================
        Configuration:
            1. Min-max pooling
            5. Only pooling
            
        ===========================================================
        
        '''
        
        print('running vectorizer test 3 ...')
        
        MODEL_FILE = 'model.bin'
        model = KeyedVectors.load_word2vec_format(MODEL_FILE, binary=True)
        
        config = VConfig('gensim',
                         model = model,
                         average_type = None,
                         min_pooling = True,
                         max_pooling = True,
                         concatenation = 'pooling',
                         return_type = None)
        
        vectorizer = Vectorizer(config)
        
        case = [['отвергнуть_VERB', 'щедрый_ADJ', 'дар_NOUN'],
            ['покупать_VERB', 'преданность_NOUN', 'дар_NOUN', 'награда_NOUN'],
            ['яд_NOUN', 'последний_ADJ', 'дар_NOUN', 'мой_DET', 'изора_NOUN']]
        
        expected = pickle.load(open('./sanity_checks_data/vectorizer3.p', 'rb'))
        
        res = vectorizer.fit(case).transform(case)
        
        np.testing.assert_allclose(expected, res)
        
        print('vectorizer test 3 is passed')
        
    def test4():
        '''
        ===========================================================
        Configuration:
            1. Simple average
            2. Max pooling
            3. Return average + max pooling
            
            
        ===========================================================
        
        '''
        
        print('running vectorizer test 4 ...')
        
        MODEL_FILE = 'model.bin'
        model = KeyedVectors.load_word2vec_format(MODEL_FILE, binary=True)
        
        config = VConfig('gensim',
                         model = model,
                         average_type = 'simple_average',
                         min_pooling = False,
                         max_pooling = True,
                         concatenation = 'full',
                         return_type = None)
        
        vectorizer = Vectorizer(config)
        
        case = [['отвергнуть_VERB', 'щедрый_ADJ', 'дар_NOUN'],
            ['покупать_VERB', 'преданность_NOUN', 'дар_NOUN', 'награда_NOUN'],
            ['яд_NOUN', 'последний_ADJ', 'дар_NOUN', 'мой_DET', 'изора_NOUN']]
        
        expected = pickle.load(open('./sanity_checks_data/vectorizer4.p', 'rb'))
        
        res = vectorizer.fit(case).transform(case)
        
        np.testing.assert_allclose(expected, res)
        
        print('vectorizer test 4 is passed')
        
    def test5():
        '''
        ===========================================================
        Configuration:
            1. Return only min pooling
            
            
        ===========================================================
        
        '''
        
        print('running vectorizer test 5 ...')
        
        MODEL_FILE = 'model.bin'
        model = KeyedVectors.load_word2vec_format(MODEL_FILE, binary=True)
        
        config = VConfig('gensim',
                         model = model,
                         average_type = None,
                         min_pooling = True,
                         max_pooling = False,
                         concatenation = None,
                         return_type = 'only_min_pooling')
        
        vectorizer = Vectorizer(config)
        
        case = [['отвергнуть_VERB', 'щедрый_ADJ', 'дар_NOUN'],
            ['покупать_VERB', 'преданность_NOUN', 'дар_NOUN', 'награда_NOUN'],
            ['яд_NOUN', 'последний_ADJ', 'дар_NOUN', 'мой_DET', 'изора_NOUN']]
        
        expected = pickle.load(open('./sanity_checks_data/vectorizer5.p', 'rb'))
        
        res = vectorizer.fit(case).transform(case)
        
        np.testing.assert_allclose(expected, res)
        
        print('vectorizer test 5 is passed')
        
    def test6():
        '''
        ===========================================================
        Configuration:
            1. Full
            2. One word in line
            
            
        ===========================================================
        
        '''
        
        print('running vectorizer test 6 ...')
        
        MODEL_FILE = 'model.bin'
        model = KeyedVectors.load_word2vec_format(MODEL_FILE, binary=True)
        
        config = VConfig('gensim',
                         model = model,
                         average_type = 'simple_average',
                         min_pooling = True,
                         max_pooling = True,
                         concatenation = 'full',
                         return_type = None)
        
        vectorizer = Vectorizer(config)
        
        case = [['отвергнуть_VERB', 'щедрый_ADJ'],
                ['покупать_VERB']]
        
        expected = pickle.load(open('./sanity_checks_data/vectorizer6.p', 'rb'))
        
        res = vectorizer.fit(case).transform(case)
        
        np.testing.assert_allclose(expected, res)
        
        print('vectorizer test 6 is passed')
        
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
        
        

def main():
    
    run_preprocessor_unittests()
    
    print('all preprocessor tests are passed')
    
    run_vectorizer_unittests()
    
    print('all vectorizer tests are passed')
    
    print('ALL TESTS ARE PASSED')
    
    
if __name__ == '__main__':
    
    main()
    
        
        
        
        