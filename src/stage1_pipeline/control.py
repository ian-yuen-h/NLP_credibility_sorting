from elastic_search import esearch
from webscrape import extract_text
from sentiment_analysis import extract_sentiment, filter_outliers, do_plot


def main():
    results = esearch()
    res_dict = extract_save(results)
    res_dict = extract_text(res_dict)
    res_dict, subjectivity_vals, polarity_vals, id_legend = extract_sentiment(res_dict)

    pass



if __name__ == "__main__":
    main()