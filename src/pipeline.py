import configparser
from pathlib import Path
from elasticsearch import Elasticsearch

##  Configuration variables
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / 'uchicago.ini'


def initialize_elastic_search():
    """ Sets up connection to server and returns elastic search object """
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    server_config = config['server']
    es = Elasticsearch(
        [{ 'host': server_config['host'], 'port': server_config['port'] }],
        http_auth=(server_config['username'], server_config['password']))
    return es


if __name__ == "__main__":
    es = initialize_elastic_search()
    res = es.search(
        index='emm_data',
        size=1,
        body={ "query": { "match_all": {} } },
        _source=['source', 'link']
    )
    print(res)