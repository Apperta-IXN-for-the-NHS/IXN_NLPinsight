#########################################################################
# This method has been adapted from the following source
# Link: https://shuaiw.github.io/2016/12/22/topic-modeling-and-tsne-visualzation.html
# Title: Topic Modeling and t-SNE Visualization
# Author: Shuai
# Date: 22/12/2016
#########################################################################

import os
import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import bokeh.plotting as bp
from bokeh.models import HoverTool
from bokeh.resources import CDN
from bokeh.embed import file_html
import random
from flask import session
import pandas as pd
from stopwords import stop_word_list
import pickle
import lda
import flask
from flask import session


def lda_tsne(total_text, file_names, n_topics=None, n_top_words=None):
    '''
    Handles the process of applying Latent Dirichlet Allocation (LDA) to the input text and the 
    dimensionality reduction of the result from LDA using t-SNE.
    The LDA algorithm returns a document-topic probability matrix which describes the probabilities of 
    the topics in a document. 
    The result of t-SNE are x,y coordinates that can be plotted on a scatter plot to visualise the clusters. 
    The bokeh library is used for visualisation, it provides great interactive plots and hovertools that can 
    add extra information to the plot.
    It also create an html output 
    that can be easily embedded within a web page. 
    
    Parameters
    ----------
    tota_text : list
        A list of strings where each element is all the text of a  document in one string.
    file_names : list
        NA list of strings where each element is a file name of the files that were uploaded.
    n_topics: int
        This is a hyperparameter of the sklearn LDA function, it needs to know how many topics are being modelled. 
    n_top_words: int
        This is a number representing the number of words that described each topic. This is used on the bokeh hover tool. 

    Returns
    ----------
    html : str
        the html embedding of the bokeh plot, this can be directly embedded in a web page. 

    '''

    #loads the flask session variable in order to use it when object serialisations (pickle) to the filing system.
    myid = session['myid']

    n_data = len(file_names)

    #if the number of topics is not specified (like when a user first launches the clusterinfg), 
    # it uses a rule of thumb to estimate the number of topics in a corpus of documents
    #the rule of thumb is ((number of documents)/2)^0.5.
    #another option is to use a more advanced algorithm to estimate the number of topics. 
    # I have tried HDBSCAN but the result is highly dependent on its 'minimal cluster size' parameter.
    if n_topics is None:
        n_topics = int(round(((len(file_names))/2)**0.5))
        session['number_topics'] = str(n_topics)

    # if the number of top words is not specific, use 5 words to described a topic
    if n_top_words is None:
        n_top_words = 5
        session['number_topwords'] = str(n_top_words)

    #the timing is for testing, to see how long it takes to run certain functions.
    t0 = time.time()

    #loads the list of stop words
    stopwords = stop_word_list()

    #loads the Scikit-Learn countvectorizer. This will convert the input text into a document-term matrix.
    #It is a matrix that simply registers a count of the different n-grams within the text
    #When the ngram_range paramters is set to (1,1) the ngrams are only the different words without a documents.
    # so for the sentence "My name is David" the list of ngrams would be ['My', 'name', 'is', 'david']
    # if the ngram_range parameter is set to (1,2) it will also include bigrams
    # for the same sentence the ngrams would be ['My', 'My name', 'name', 'name is', 'is', 'is david', 'david']
    cvectorizer = CountVectorizer(
        min_df=1, stop_words=stopwords,  lowercase=True, ngram_range=(1, 3))

    # this creates the document-term matrix
    cvz = cvectorizer.fit_transform(total_text)

    t1 = time.time()

    print("Time for count vectorizer (document term matrix): " + str(t1-t0))

    t2 = time.time()
    # generates the lda model with 500 iterations
    lda_model = lda.LDA(n_topics, 500)

    # fits the lda model to the document-term matrix
    X_topics = lda_model.fit_transform(cvz)

    t3 = time.time()

    print("Time for LDA: " + str(t3-t2))

    if not os.path.exists('pickles'):
        os.makedirs('pickles')

    # creates the paths to which the pickled objects will be saved
    lda_model_path = "pickles/lda_model_" + str(myid) + '.p'
    document_term_matrix_path = "pickles/document_term_matrix_" + \
        str(myid) + '.p'
    cvectorizer_path = "pickles/cvectorizer_" + str(myid) + '.p'

    #pickles the objects and saves them
    pickle.dump(lda_model, open(lda_model_path, "wb"))
    pickle.dump(cvz, open(document_term_matrix_path, "wb"))
    pickle.dump(cvectorizer, open(cvectorizer_path, "wb"))

    #the number of files uploaded
    num_example = len(X_topics)

    t4 = time.time()

    #creates the t-SNE object that will be used, the number of components reffers to the number of output dimensions
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.2,
                      init='pca')

    #uses t-SNE to calculate the 2-D coordinates representing the documents.
    tsne_lda = tsne_model.fit_transform(X_topics[:num_example])

    t5 = time.time()

    print("Time for TSNE: " + str(t5-t4))

    #Some fancy processing of the data using pandas to remove any 'NAN' values from the data.
    tsne_lda_df = pd.DataFrame(tsne_lda)

    print(tsne_lda_df.describe())

    tsne_lda_df = tsne_lda_df.fillna('')

    tsne_lda = tsne_lda[~np.isnan(tsne_lda).any(axis=1)]

    tsne_lda_df = tsne_lda_df[~tsne_lda_df.isin(
        [np.nan, np.inf, -np.inf]).any(1)]

    print(tsne_lda_df.describe())

    # finds the most probable topic for each document and saves it into the list
    _lda_keys = []
    for i in range(X_topics.shape[0]):
        _lda_keys += X_topics[i].argmax(),

    #gets the most probable words of each topic as a representaiton of that topic.
    topic_summaries = []
    topic_word = lda_model.components_  # get the topic words
    vocab = cvectorizer.get_feature_names()
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(
            topic_dist)][:-(n_top_words+1):-1]
        topic_summaries.append(' '.join(topic_words))

    #creates a colourmap to colour each topic in a separate randomly chosen colour
    colormap = np.array([])

    for i in range(n_topics):
        color = "#" + "%06x" % random.randint(0, 0xFFFFFF)
        colormap = np.append(colormap, color)

    raw_topic_summaries = []
    for x in _lda_keys:
        raw_topic_summaries.append(topic_summaries[x])

    t6 = time.time()
    title = " t-SNE visualization of LDA model trained on {} files, " \
            "{} topics, {} data " \
            "points and top {} words".format(
                X_topics.shape[0], n_topics, num_example, n_top_words)

    #creates the bokeh figure objects that will be used to crate the sactter plot
    plot_lda = bp.figure(plot_width=1200, plot_height=700,
                         title=title,
                         tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                         x_axis_type=None, y_axis_type=None, min_border=1)

    # defines the size of the plot dots, the more there are of them the smaller they should be
    if n_data < 30:
        dot_size = 20
    if n_data >= 30 and n_data < 50:
        dot_size = 15
    if n_data >= 50 and n_data < 150:
        dot_size = 11
    if n_data >= 150:
        dot_size = 5

    #this object defines the paramters of the plot in the form of a dictionary. The file_names and raw_topic_summaries are used
    #for the plot's hover tool.
    source = bp.ColumnDataSource(data=dict(x=tsne_lda_df.iloc[:, 0], y=tsne_lda_df.iloc[:, 1],
                                           color=colormap[_lda_keys][:num_example], file_names=file_names, 
                                           raw_topic_summaries=raw_topic_summaries))
    plot_lda.scatter(x='x', y='y',
                     color='color',
                     source=source, size=dot_size)

    plot_lda.outline_line_width = 7
    plot_lda.outline_line_alpha = 0.3
    plot_lda.outline_line_color = "#353A40"

    # randomly choses a file as the coordinate at which to show the topic words.
    topic_coord = np.empty((X_topics.shape[1], 2)) * np.nan
    for topic_num in _lda_keys:
        if not np.isnan(topic_coord).any():
            break
        topic_coord[topic_num] = tsne_lda[_lda_keys.index(topic_num)]

    # plots the top words
    for i in range(X_topics.shape[1]):
        plot_lda.text(topic_coord[i, 0], topic_coord[i, 1], [
                      topic_summaries[i]])

    #sets the bokeh's hover tool to display the file name and topic summary of 
    # a dot when the cursor hovers over a dot.
    hover = plot_lda.select(dict(type=HoverTool))
    hover.tooltips = [("file name", "@file_names"),
                      ("topic summary", '@raw_topic_summaries')]

    t7 = time.time()
    print("Time for Bokeh plotting: " + str(t7-t6))

    print('\n>>> whole process done; took {} mins\n'.format((t7 - t0) / 60.))

    #creates the html code of the visualisation that will be used in the html template.
    html = file_html(plot_lda, CDN)

    #pickles and saves the objects for later use
    raw_topic_summaries_path = "pickles/raw_topic_summaries" + str(myid) + '.p'
    lda_keys_path = "pickles/lda_keys_path" + str(myid) + '.p'

    pickle.dump(raw_topic_summaries, open(raw_topic_summaries_path, "wb"))
    pickle.dump(_lda_keys, open(lda_keys_path, "wb"))

    return html
