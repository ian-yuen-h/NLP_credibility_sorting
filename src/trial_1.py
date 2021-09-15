from elasticsearch import Elasticsearch

import json

import newspaper
from newspaper import Article

import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

import scipy
import numpy as np
import matplotlib.pyplot as plt




def fetch_elastic_data():
    es = Elasticsearch([{'host':'localhost', 'port':9200}], http_auth=('uchicago', '3fMy7Wq90LDWBCMEXt6j'))
    res = es.search(
        index='emm_data',
        size=5000,
        body={'query':{'match_all':{}}},
        _source=['source', 'link']
    )
    return res['hits']['hits']


#
# # store article's id and url in json
# def get_id_and_url(res):
#     results = res['hits']['hits']
#     res_dict = {}
#     for r in results:
#         # get result id and url
#         res_id = r['_id']
#         res_url = r['_source']['link']
#
#         # store the values in a dictionary
#         res_dict[res_id] = res_url
# #     j = json.dumps(res_dict)
#     # save info to a json file
#     with open('data.json', 'w') as f:
#         json.dump(res_dict, f)
#
# get_id_and_url(res)
#
# # get and return a list of urls
# def get_urls():
#     results = res['hits']['hits']
#     urls = {}
#     for u in results:
#         url_id = u['_id']
#         url = u['_source']['link']
#         urls[url_id] = url
#     return urls
#
# # len(get_urls())
# get_urls()
#
#
#
# # get article's url
# url = res['hits']['hits'][0].get('_source')['link']
#
# # create Article object
# article = Article(url)
#
# # download & parse the article
# article.download()
# article.parse()
#
# # extract NL properties from the text
# article.nlp()
#
# # get author(s), publish date & top image
# f'author: {article.authors},    publication date: {article.publish_date},    top image: {article.top_image}'
#
# # get the article's text
# article.text
#
# # now perform NLP on the article's text with spaCy
#
# nlp = spacy.load('en_core_web_sm')
#
# # store the article's text
# article_text = article.text
#
# # tokenize the text
# article_doc = nlp(article_text)
#
# # extract tokens from the doc
# article_tokens = [token.text for token in article_doc]
#
# # print the tokens
# article_tokens
#
# # extract info and perfom NLP on articles
# def extract_article_info():
#     articles_dict = {}
# #     for i in range(5):
#     for url_id, url in get_urls().items()[:5]:
#         # create Article object
#         article = Article(url)
#
#         # check for valid url
# #             if article.is_valid_url():
#         try:
#             # download and parse the article
#             article.download()
#             article.parse()
#             # extract NL properties from article's text
#             article.nlp()
#             # store article's text
#             articles_dict[url_id] = article.text
# #                 print(f'{url_id} : {article.text}')
#         except:
#             print('could not download article')
#     return articles_dict
#
# # articles_dict = extract_article_info()
#
# print(extract_article_info().items())
#
#
#
# nlp = spacy.load('en_core_web_sm')
# nlp.add_pipe('spacytextblob')
#
# articles_attr = {}
# subjectivity_vals = np.empty(shape=[0, 1])
# polarity_vals = np.empty(shape=[0, 1])
# id_legend = np.empty(shape=[0, 1])
#
# for key, value in articles_dict.items():
#     doc = nlp(each)
#     articles_attr[key] = [doc._.polarity, doc._.subjectivity, doc._.assessments]
#     np.append(subjectivity_vals, doc._.subjectivity)
#     np.append(polarity_vals, doc._.polarity)
#     np.append(id_legend, key)
#
# #subjectivity: 0.0 = objective, 1.0 = subjective
# #polarity: measures level of approval/disapproval
# #assessments = list of polarity, subjectivity scores for assessed tokens
#
#
# #x-y plot of subjectivity, polarity
# #see if there is any relationship, clustering
#
#
#
# def do_plot(subjectivity_vals, polarity_vals, tags):
#     # Plot
#     plt.plot(subjectivtity_vals, polarity_vals, label="oridinal data")
#     plt.xlabel("Subjectivity")
#     plt.ylabel("Polarity")
#     plt.title("Subjectivity-Polarity Word-level")
#     plt.legend()
#     m, b = np.polyfit(x, y, 1) #regression
#     slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
#     plt.plot(x, m*x + b)
#     plt.savefig("initial_plot.png")
#     plt.close()
#
# do_plot(subjectivity_vals, polarity_vals, id_legend)
#

def main():
    result = fetch_elastic_data()
    print(result)

if __name__ == '__main__':
    main()
