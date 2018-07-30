import pandas as pd
pd.options.mode.chained_assignment = None 
import numpy as np
import re
import nltk
import io
import base64
from gensim.models import word2vec

from sklearn.manifold import TSNE
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stopwords import stop_word_list

import time

from wordcloud import WordCloud, STOPWORDS




def build_word_cloud(token_list, n, myextension):
    words  = token_list
    stop_words = stop_word_list()

   

    

    wordcloud = WordCloud(width=1440, height=1080,
                          background_color='white',
                         #colormap="Blues",
                         #margin=10,
                          stopwords=stop_words,
                          max_words=n,
                         ).generate(str(words))



    fig = plt.figure(figsize=(20, 15))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.margins(x=0, y=0)
    
    fig_name='static/mycloud' + str(myextension)

    plt.savefig(fig_name, bbox_inches='tight')
  


