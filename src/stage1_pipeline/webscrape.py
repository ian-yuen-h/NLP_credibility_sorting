import newspaper
from newspaper import Article


def extract_text(res_dict):
    errored_articles = []
    for key, value in res_dict.items():
        target_url = value[0]
        article = Article(target_url)
        if article.is_valid_url():
            article.download()
        try:    
            article.parse()
        except:
            errored_articles.append(key)
            continue
        res_dict[key].append(article.text)

    with open('errored_articles.txt', 'w') as f:
        f.write("Articles that cannot be scraped by uid: \n")
        for each in errored_articles:
            f.write(each)

    return res_dict

def main():
    pass

if __name__ == "__main__":
    main()