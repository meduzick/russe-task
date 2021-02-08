# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 12:07:02 2021

@author: User
"""
import re
from unicodedata import normalize
from collections import namedtuple

class Config(namedtuple('preprocessor_config',
                        ['regexp',
                         'stopwords',
                         'analyzer',
                         'with_pos_tag',
                         'remove_stop_words',
                         'lemmatize',
                         'tokenizer'])):
    '''
    =============================================================
    Embodies configuration properties for preprocessor.
    Args:
        1. regexp (str): regular expression string
        2. stopwords (set): set of stopwords
        3. analyzer (Maru analyzer): used to perform lemmatization
        4. with_pos_tag (bool): whether append pos tag or not
        5. remove_stop_words (bool): whether to remove stop words
                                    or not
        6. lemmatize (bool): whether to perform lemmatization or
                            not
        7. tokenizer (razdel): used to make tokenization
    =============================================================
    '''
    pass

class Preprocessor():
    '''
    =============================================================
    Takes care about preprocessing steps.
    Preprocessing steps can be as follows:
        1. Lowercasing
        2. Unicode normalization
        3. Cleaning out (remove unnecessary symbols)
        4. Remove stop words
        5. Text normalization
        6. POS tagging
    =============================================================
    '''
    def __init__(self, config):
        self._regexp = config.regexp
        self._stopwords = config.stopwords
        self._lemmatize = config.lemmatize
        self._lemma_analyzer = config.analyzer
        self._with_pos_tag = config.with_pos_tag
        self._remove_stopwords = config.remove_stop_words
        self._tokenizer = config.tokenizer
        
    @property
    def lemma_analyzer(self):
        return self._lemma_analyzer
    
    def fit(self, X, y = None):
        '''
        ==========================================================
        Does nothing
        ==========================================================
        '''
        return self
    
    def transform(self, X, y = None):
        '''
        ===========================================================
        Execute preprocessing steps.
        Args:
            X (list): list of strings
            y (None): ----
        Returns:
            List of lists of preprocessed strings
        ===========================================================
        '''
        res = []
        for line in X:
            line = re.sub(self._regexp, ' ', normalize('NFKC', line.lower()))
            # Remove unnecessary gaps
            line = re.sub(r'[ ]{2,}', ' ', line).strip(' ')
            if self._tokenizer is not None:
                assert hasattr(self._tokenizer, 'tokenize'), \
                    'tokenizer does have tokenize method'
                line = [token.text for token in self._tokenizer.tokenize(line)]
            else:
                line = line.split()
            if self._remove_stopwords:
                line = [word for word in line if word not in self._stopwords]
            if self._with_pos_tag:
                line = [morf.lemma + '_' + morf.tag.pos \
                        for morf in self._lemma_analyzer.analyze(line)]
            else:
                line = [morf.lemma \
                        if self._lemmatize \
                        else morf.word \
                        for morf in self._lemma_analyzer.analyze(line)]
            res.append(line)
        return res
    