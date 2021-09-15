import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


def extract_sentiment(res_dict):
    """res_dict[uid] = [url, publisher, text, polarity, subjectivity, assessments]"""
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('spacytextblob')
    articles_attr = {}
    subjectivity_vals = np.empty(shape=[0, 1])
    polarity_vals = np.empty(shape=[0, 1])
    id_legend = np.empty(shape=[0, 1])
    publishers_vals = np.empty(shape=[0, 1])
    publisher_hashes = {}
    for key, value in res_dict.items():
        doc = nlp(value)
        res_dict[key].append(doc._.polarity, doc._.subjectivity, doc._.assessments)
        subjectivity_vals = np.append(subjectivity_vals, doc._.subjectivity)
        polarity_vals = np.append(polarity_vals, doc._.polarity)
        id_legend = np.append(id_legend, key)
    

    return res_dict, subjectivity_vals, polarity_vals, id_legend

def filter_outliers():
    pass

def do_plot(subjectivity_vals, polarity_vals, tags):
    # Plot
    plt.scatter(subjectivity_vals, polarity_vals, label="oridinal data")
    plt.xlabel("Subjectivity")
    plt.ylabel("Polarity")
    plt.title("Subjectivity-Polarity Word-level")
    plt.legend()
    m, b = np.polyfit(subjectivity_vals, polarity_vals, 1) #regression
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(subjectivity_vals,  polarity_vals)
    plt.plot(subjectivity_vals, m*subjectivity_vals + b)
    plt.savefig("initial_plot.png")
    plt.close()



def main():
    pass



if __name__ == "__main__":
    main()