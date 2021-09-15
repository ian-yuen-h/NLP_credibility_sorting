from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import json

def esearch():
    """Elastic Search Function"""
    es = Elasticsearch([{'host':'localhost', 'port':9200}], http_auth=('uchicago', '3fMy7Wq90LDWBCMEXt6j'))
    
    res = scan(
        es,
        index='credibility_training_data', 
        size=5000, 
        query={'query':{'match_all':{}}}
    )

    res_dict = {}
    for each in res:
        uid = each['_id']
        url = each['_source']['link']
        credibility = each['_source']["credibility"]
        domain = each['_source']["registered_domain"]
        try:
            text = each['_source']["text"]
            source = each['_source']['source']
        except:
            continue
        res_dict[uid]={"url": url, "source": source, "credibility": credibility, "domain": domain, "text": text}
    
    with open('data.json', 'w') as f:
        json.dump(res_dict, f)

    return results

# def extract_save(results):
#     """Extract id, url, source"""
#     res_dict = {}
#     for r in results:
#         res_id = r['_id']
#         res_url = r['_source']['link']
#         res_publisher = r['_source']['_source']
#         res_dict[res_id] = [res_url, res_publisher]
#         #add credibility
        
#     with open('data.json', 'w') as f:
#         json.dump(res_dict, f)
    
#     return res_dict



def main():
    pass



if __name__ == "__main__":
    main()