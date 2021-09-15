import configparser
from pathlib import Path
from elasticsearch import Elasticsearch

##  Configuration variables
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / 'uchicago.ini'


def initialize_elastic_search():
    """ Sets up connection to server and returns elastic search object 
    
    Note: To successfully run, follow setup instructions in README.
    """
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    server_config = config['server']
    es = Elasticsearch(
        [{ 'host': server_config['host'], 'port': server_config['port'] }],
        http_auth=(server_config['username'], server_config['password']))
    return es

def fetch_elastic_data(dataset, n=None):
    """ Fetch article data from elastic search 
    
    Args:
        dataset (str): Name of dataset to fetch data from
        n (int): Number of articles to fetch. If None, all articles
            will be fetched
    Returns:
        dictionary mapping article id to article metadata (including 
        credibility score if dataset is credibility_training_data) 
    """
    pass

def fetch_article_text(articles_dictionary):
    """ Retrieves article text based on article urls 
    
    Args:
        articles_dictionary (dict): dictionary mapping article id to article
            metadata
    """