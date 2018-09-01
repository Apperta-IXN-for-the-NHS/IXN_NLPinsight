import nltk
from nltk import FreqDist
import pygal


def frequency_dist(data, n, title, stopwords=None):
    '''
    Computes the word-frequency distribution graph and returns a base 64 encoded data uri version of it.
    

    Parameters
    ----------
    data : list
        List of text tokens.

    Returns
    ----------
    Word-frequency distribution graph in base 64 encoded data uri.
    '''

    if stopwords is not None:
        data = [x for x in data if not x in stopwords]

    data = [word[:-2] if word.lower().endswith("'s") else word
            for word in data]

    dist = FreqDist(data)

    most_common = dist.most_common(n)

    words_ordered = []
    frequencies_ordered = []
    for word in most_common:
        words_ordered.append(word[0])
        frequencies_ordered.append(word[1])

    line_chart = pygal.HorizontalBar()
    line_chart.title = title

    for i in range(len(words_ordered)):
        line_chart.add(words_ordered[i], frequencies_ordered[i])
    return line_chart.render_data_uri()
