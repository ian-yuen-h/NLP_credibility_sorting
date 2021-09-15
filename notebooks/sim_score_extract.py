"""
functions that relate to extracing features regarding similarity scores from given articles
"""
import warnings
import gensim
import pandas as pd
import numpy as np
import spacy
import nltk
import en_core_web_sm
from spacy import displacy
from spacytextblob.spacytextblob import SpacyTextBlob
from statistics import mean
import json
from datetime import datetime
from gensim import corpora, models, similarities


nltk.download('punkt')
warnings.filterwarnings('ignore')

NOISY_POS_TAG = ["\n\n"] 
MIN_TOKEN_LENGTH = 2


def cal_similarity_score(data):
    '''
    Calculate similarity scores with given articles
    '''
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    article_dict = {}
    article_lst = []

    for uid, info in data.items():
        article_dict[uid] = {}
        text = info['text']
        doc, token_lst = tokenize_text(text)
        article_dict[uid]["token"] = token_lst
        article_lst.append(token_lst)

    dictionary = corpora.Dictionary(article_lst)
    corpus = [dictionary.doc2bow(article) for article in article_lst]
    tfIdf_model = models.TfidfModel(corpus)

    index = similarities.SparseMatrixSimilarity(tfIdf_model[corpus], num_features=len(dictionary.keys()))
    for uid, info in article_dict.items():
        compared_bow = dictionary.doc2bow(info['token'])
        sim = index[tfIdf_model[compared_bow]]
        article_dict[uid]["sim_score"] = sim

    output_dict = {}
    for uid, info_dict in article_dict.items():
        output_dict[uid] = info_dict['sim_score']

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    return output_dict


def extract_sim_features(data_dict):
    '''
    Extract similarity score features with sim score full data dictionary
    '''
    feature_dict = {}
    count = 0
    for article_id, info in data_dict.items():
        info_new = np.delete(info, [count]) # delete the similarity score of itself
        feature_dict[article_id] = {}  
        feature_dict[article_id]["mean"] = np.mean(info_new)
        feature_dict[article_id]["median"] = np.median(info_new)
        feature_dict[article_id]["std"] = np.std(info_new)        
        feature_dict[article_id]["max"] = np.max(info_new)
        feature_dict[article_id]["min"] = np.min(info_new)
        feature_dict[article_id]["80_perc"] = np.percentile(info_new, 80)
        feature_dict[article_id]["60_perc"] = np.percentile(info_new, 60)
        feature_dict[article_id]["40_perc"] = np.percentile(info_new, 40)    
        feature_dict[article_id]["20_perc"] = np.percentile(info_new, 20)
        feature_dict[article_id]["num_over0.5"] = sum(info_new>0.5)
        feature_dict[article_id]["num_over0.75"] = sum(info_new>0.75)
        count += 1
    return feature_dict


def tokenize_text(text):
    '''
    With given text, return a list of tokenized words after removing noisy words

    Input:
      text: string
    Output:
      doc: a Spacy Doc object
      token_lst: list of tokenized words 
    '''
    nlp = en_core_web_sm.load()
    doc = nlp(text)
    token_lst = [cleanup(w.text) for w in doc if not isNoise(w)]
    return doc, token_lst


def isNoise(token):
    '''
    Check if a specific token is a noisy one
    Input:
      token: string
    Output:
      is_noise: boolean
    '''
    is_noise = False
    if token.pos_ in NOISY_POS_TAG:
        is_noise = True 
    elif token.is_stop:
        is_noise = True
    elif token.is_punct:
        is_noise = True
    elif len(token.text) <= MIN_TOKEN_LENGTH:
        is_noise = True
    return is_noise


def cleanup(token, lower = True):
    '''
    Clean up a specific token
    Input:
      token: string
    Output:
      string
    '''
    if lower:
        token = token.lower()
    return token.strip()
