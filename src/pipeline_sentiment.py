import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import numpy as np
import scipy
import matplotlib.pyplot as plt


def sentiment_attr(articles_dict):
    """
    subjectivity: 0.0 = objective, 1.0 = subjective
    polarity: measures level of approval/disapproval
    assessments = list of polarity, subjectivity scores for assessed tokens
    """
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('spacytextblob')

    articles_attr = {}
    subjectivity_vals = np.empty(shape=[0, 1])
    polarity_vals = np.empty(shape=[0, 1])
    id_legend = np.empty(shape=[0, 1])

    for key, value in articles_dict.items():
        doc = nlp(value)
        articles_attr[key] = [doc._.polarity, doc._.subjectivity, doc._.assessments]
        np.append(subjectivity_vals, doc._.subjectivity)
        np.append(polarity_vals, doc._.polarity)
        np.append(id_legend, key)


def plot_sentiment_attr(subjectivity_vals, polarity_vals, tags):
    """
    x-y plot of subjectivity, polarity
    see if there is any relationship, clustering
    """
    plt.plot(subjectivity_vals, polarity_vals, label="oridinal data")
    plt.xlabel("Subjectivity")
    plt.ylabel("Polarity")
    plt.title("Subjectivity-Polarity Word-level")
    plt.legend()
    m, b = np.polyfit(x, y, 1) #regression
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(subjectivity_vals, polarity_vals)
    plt.plot(x, m*x + b)
    plt.savefig("initial_plot.png")
    plt.close()
    with open('textblob_statistics.txt', 'w') as outfile:
        outfile.write("Slope: ", slope)
        outfile.write("Intercept: ", intercept)
        outfile.write("r_value: ", r_value)
        outfile.write("p_value: ", p_value)
        outfile.write("std_err: ", std_err)

def main():
    sentiment_attr()
    pass


if __name__ == "__main__":
    main()
