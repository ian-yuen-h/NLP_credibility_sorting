import warnings
import gensim
import pandas as pd
import numpy as np
import sklearn
import spacy
import nltk
import en_core_web_sm
import wordcloud
from spacy.symbols import ORTH
from collections import Counter
from spacy import displacy
import matplotlib.pyplot as plt
import re
from textacy import preprocessing
from textacy import extract as ex
from spacytextblob.spacytextblob import SpacyTextBlob
from statistics import mean
import json
from datetime import datetime
from gensim import corpora, models, similarities


nltk.download('punkt')
warnings.filterwarnings('ignore')

NOISY_POS_TAG = ["\n\n"] 
MIN_TOKEN_LENGTH = 2


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


def find_common_token(text):
    '''
    Find most common token with given text
    Input:
      text: string
    Output:
      common_word_lst: list of tuples (token, times)
    '''
    token_lst = tokenize_text(text)
    common_word_lst = Counter(cleaned_lst).most_common(10)
    return common_word_lst


def detect_entity(doc):
    '''
    Detect entities with given document
    Input:
      doc: a Spacy Doc object
    Output:
      entity_dict: dictionary of entities
      entity_num: number of entities
      entity_type_num: number of entity types
    '''
    entity_dict = {}
    labels = set([w.label_ for w in doc.ents])
    entity_num = 0
    for label in labels: 
        entities = [cleanup(e.text, lower=False) for e in doc.ents if label==e.label_] 
        entities = list(set(entities))
        entity_num += len(entities)
        entity_dict[label] = entities
    entity_type_num = len(entity_dict)
    return entity_dict, entity_num, entity_type_num


def display_entity(doc):
    '''
    Detect entities with given document
    Input:
      doc: a Spacy Doc object
    '''
    displacy.render(doc, style='ent', jupyter=True)
    
    
def draw_word_cloud(text):
    '''
    Draw word cloud
    '''
    wc = wordcloud.WordCloud(background_color="white", max_words=500, width= 1000, height = 1000, mode ='RGBA', scale=.5).generate(text)
    plt.imshow(wc)
    plt.axis("off")
    plt.savefig("whitehouse_word_cloud.pdf", format = 'pdf')


def get_sentences(text):
    '''
    Returns list of sentences
    '''
    doc = nlp(text)
    sentences = list(doc.sents)
    return sentences


def get_quotes(text):
    '''
    Return quotations based on given text
    Output:
        quotes: class textacy.extract.triples.DQTriple
        https://textacy.readthedocs.io/en/latest/api_reference/extract.html?highlight=extract#textacy.extract.triples.DQTriple 
    '''
    # quotes = re.findall(r'“(.*?)“',text) # r'^“.*”$' | 
    # return quotes
    nlp = en_core_web_sm.load()
    text = preprocessing.normalize.quotation_marks(text)
    doc = nlp(text)
    quotes = ex.direct_quotations(doc)
    return quotes


def get_json_data(filepath):
    # Opening JSON file
    f = open(filepath,) 
    # returns JSON object as 
    # a dictionary
    data = json.load(f)
    # Closing file
    f.close()
    return data


def get_sentiment(text):
    '''
    subjectivity: 0.0 = objective, 1.0 = subjective
    polarity: measures level of approval/disapproval
    '''
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('spacytextblob')
    doc = nlp(text)
    subjectivity = doc._.subjectivity
    polarity = doc._.polarity
    return subjectivity, polarity


def get_quotation_sentiment(text):
    '''
    Get the number of direct quotations,
    average subjectivity score and polarity score of quotations, None if no direct quotations
    '''
    quotes = get_quotes(text)
    sub_score_lst = []
    pol_score_lst = []
    num_quote = 0
    for item in quotes:
        item_text = str(item.content) #get quotation text
        subjectivity, polarity = get_sentiment(item_text)
        sub_score_lst.append(subjectivity)
        pol_score_lst.append(polarity)
        num_quote += 1
    if num_quote == 0:
        return num_quote, None, None
    else:
        return num_quote, mean(sub_score_lst), mean(pol_score_lst)


def get_training_features():
    '''
    Pipeline of extracting entity and quotation features from dataset
    '''
    data = get_json_data('data.json')
    result_dict = {}

    for uid, info in data.items():
        text = info['text']
        doc, token_lst = tokenize_text(text)
        entity_dict, entity_num, entity_type_num = detect_entity(doc)
        try:
            num_quote, quote_subjectivity, quote_polarity = get_quotation_sentiment(text)
        except:
            num_quote, quote_subjectivity, quote_polarity = None, None, None
        result_dict[uid] = {}
        result_dict[uid]['entity_dict'] = entity_dict
        result_dict[uid]['entity_num'] = entity_num
        result_dict[uid]['entity_type_num'] = entity_type_num
        result_dict[uid]['num_quote'] = num_quote
        result_dict[uid]['quote_subjectivity'] = quote_subjectivity
        result_dict[uid]['quote_polarity'] = quote_polarity
        
    return result_dict


def get_training_features_smalltrial():
    '''
    Pipeline of extracting entity and quotation features from dataset
    '''
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    data = get_json_data('data.json')
    result_dict = {}
    count = 0 #
    for uid, info in data.items():
        
        count += 1 ##
        text = info['text']
        doc, token_lst = tokenize_text(text)
        entity_dict, entity_num, entity_type_num = detect_entity(doc)
        try:
            num_quote, quote_subjectivity, quote_polarity = get_quotation_sentiment(text)
        except:
            num_quote, quote_subjectivity, quote_polarity = None, None, None
        result_dict[uid] = {}
        result_dict[uid]['entity_dict'] = entity_dict
        result_dict[uid]['entity_num'] = entity_num
        result_dict[uid]['entity_type_num'] = entity_type_num
        result_dict[uid]['num_quote'] = num_quote
        result_dict[uid]['quote_subjectivity'] = quote_subjectivity
        result_dict[uid]['quote_polarity'] = quote_polarity
        
        if count > 50:
            break

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    return result_dict


def get_training_features_quote_smalltrial():
    '''
    Small trial only includes quotation features
    '''
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    data = get_json_data('data.json')
    result_dict = {}
    # count = 0 #
    for uid, info in data.items():
        
        # count += 1 ##
        text = info['text']
        # doc, token_lst = tokenize_text(text)
        # entity_dict, entity_num, entity_type_num = detect_entity(doc)
        try:
            num_quote, quote_subjectivity, quote_polarity = get_quotation_sentiment(text)
        except:
            num_quote, quote_subjectivity, quote_polarity = None, None, None
        result_dict[uid] = {}
        # result_dict[uid]['entity_dict'] = entity_dict
        # result_dict[uid]['entity_num'] = entity_num
        # result_dict[uid]['entity_type_num'] = entity_type_num
        result_dict[uid]['num_quote'] = num_quote
        result_dict[uid]['quote_subjectivity'] = quote_subjectivity
        result_dict[uid]['quote_polarity'] = quote_polarity
        
        # if count > 50:
        #     break

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    return result_dict


def get_training_features_entity_smalltrial():
    '''
    Small trial only includes quotation features
    '''
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    data = get_json_data('data.json')
    result_dict = {}
    count = 0 #
    for uid, info in data.items():
        
        count += 1 ##
        text = info['text']
        doc, token_lst = tokenize_text(text)
        entity_dict, entity_num, entity_type_num = detect_entity(doc)
        # try:
        #     num_quote, quote_subjectivity, quote_polarity = get_quotation_sentiment(text)
        # except:
        #     num_quote, quote_subjectivity, quote_polarity = None, None, None
        result_dict[uid] = {}
        result_dict[uid]['entity_dict'] = entity_dict
        result_dict[uid]['entity_num'] = entity_num
        result_dict[uid]['entity_type_num'] = entity_type_num
        # result_dict[uid]['num_quote'] = num_quote
        # result_dict[uid]['quote_subjectivity'] = quote_subjectivity
        # result_dict[uid]['quote_polarity'] = quote_polarity
        
        # if count > 50:
        #     break

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    return result_dict


def cal_similarity_score(data):
    '''
    Calculate similarity scores with given articles
    '''
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    # data = get_json_data('data.json')
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
