import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpld3
from stopwords import stop_word_list
from wordcloud import WordCloud


def build_word_cloud(text, n):
    '''
    Plots the wordcloud from a given token list and returns the plot in html format to be embedded in a html file.

    Parameters
    ----------
    text : str
        The text in form of a string to generate the word cloud.
    n : int:
        maximum number of tokens to display in the wordcloud.

    Returns
    ----------
    Embedded html of the wordcloud visulisation. This can be simply added to a html template.

    '''
   
    stop_words = stop_word_list()

    wordcloud = WordCloud(width=1440, height=1080,
                          background_color='white',
                          #colormap="Blues",
                          #margin=10,
                          stopwords=stop_words,
                          max_words=n,
                          ).generate(str(text))

    fig = plt.figure(figsize=(13, 9))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.margins(x=0, y=0)

    html = mpld3.fig_to_html(fig, no_extras=True, template_type='general')

    return html
