#https://shuaiw.github.io/2016/12/22/topic-modeling-and-tsne-visualzation.html
#https://github.com/ShuaiW/twitter-analysis/blob/master/topic_20news.py




import os
import argparse
import time
import lda
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import bokeh.plotting as bp
from bokeh.plotting import save, output_file, show
from bokeh.models import HoverTool
from bokeh.resources import CDN
from bokeh.embed import file_html
import random
from bokeh.embed import components
from sklearn.decomposition import LatentDirichletAllocation





def lda_tsne(total_text, file_names):

    n_data = len(file_names)

    n_topics =  int(round(((len(file_names))/2)**0.5))
    n_iter = 200
    n_top_words = 5
    threshold = 0.5

    t0 = time.time()


    cvectorizer = CountVectorizer(min_df=1, stop_words='english')
    cvz = cvectorizer.fit_transform(total_text)


    

    lda_model = lda.LDA(n_topics=n_topics, n_iter=n_iter)
    X_topics = lda_model.fit_transform(cvz)

    print(X_topics)

    t1 = time.time()
    print('\n')

    print ('LDA training done; took {} mins'.format((t1-t0)/60.))
    print('\n')


    #np.save('mednlp/lda_doc_topic_{}files_{}topics.npy'.format(
     #   X_topics.shape[0], X_topics.shape[1]), X_topics)

    #np.save('mednlp/lda_topic_word_{}files_{}topics.npy'.format(
     #   X_topics.shape[0], X_topics.shape[1]), lda_model.topic_word_)


    ##############################################################################
    # threshold and plot

    _idx = np.amax(X_topics, axis=1) > threshold  # idx of news that > threshold

    print('idx:  ' + str(_idx))
    _topics = X_topics
    print('topics:  ' + str(_topics))
    num_example = len(_topics)

    print("num_example: " + str(num_example))


    # t-SNE: 50 -> 2D
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99,
                        init='pca')
    tsne_lda = tsne_model.fit_transform(_topics[:num_example])

    print("TSNE_LDA")
    print(type(tsne_lda))
    print(tsne_lda)
    print(tsne_lda.shape)

    tsne_lda = tsne_lda[~np.isnan(tsne_lda).any(axis=1)]
    # find the most probable topic for each news
    _lda_keys = []
    for i in range(_topics.shape[0]):
        _lda_keys += _topics[i].argmax(),

    print('lda_keys:  ')
    print(_lda_keys)

    # show topics and their top words
    topic_summaries = []
    topic_word = lda_model.topic_word_  # get the topic words
    vocab = cvectorizer.get_feature_names()
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
        topic_summaries.append(' '.join(topic_words))

    


    colormap =  np.array([])

    for i  in  range(n_topics):
        color = "#" + "%06x" % random.randint(0, 0xFFFFFF)
        colormap = np.append(colormap, color)

    print("#########################################################")
    print("COLORMAP")
    print(colormap[_lda_keys][:num_example])
    print("#########################################################")   
    print("LDA KEYS")
    print(_lda_keys[:num_example])
    print("#########################################################")



    raw_topic_summaries = []
    for x in _lda_keys:
        raw_topic_summaries.append(topic_summaries[x])

    # plot
    title = " t-SNE visualization of LDA model trained on {} files, " \
            "{} topics, thresholding at {} topic probability, {} iterations ({} data " \
            "points and top {} words)".format(
        X_topics.shape[0], n_topics, threshold, n_iter, num_example, n_top_words)

    plot_lda = bp.figure(plot_width=1200, plot_height=800,
                        title=title,
                        tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                        x_axis_type=None, y_axis_type=None, min_border=1)

    if n_data < 30 :
        dot_size = 20
    if n_data>=30 and n_data<50:
        dot_size = 15
    if n_data>=50 and n_data<150:
        dot_size = 11
    if n_data>=150:
        dot_size = 5


    source = bp.ColumnDataSource(data=dict(x=tsne_lda[:, 0], y=tsne_lda[:, 1], color = colormap[_lda_keys][:num_example], file_names = file_names, raw_topic_summaries = raw_topic_summaries))
    plot_lda.scatter(x='x', y='y',
                    color='color',
                    source=source, size = dot_size)


    plot_lda.outline_line_width = 7
    plot_lda.outline_line_alpha = 0.3
    plot_lda.outline_line_color = "#353A40"

    
    # randomly choose a news (in a topic) coordinate as the crucial words coordinate
    topic_coord = np.empty((X_topics.shape[1], 2)) * np.nan
    for topic_num in _lda_keys:
        if not np.isnan(topic_coord).any():
            break
        topic_coord[topic_num] = tsne_lda[_lda_keys.index(topic_num)]

    # plot crucial words
    for i in range(X_topics.shape[1]):
        plot_lda.text(topic_coord[i, 0], topic_coord[i, 1], [topic_summaries[i]])
    



    # hover tools
    hover = plot_lda.select(dict(type=HoverTool))
    hover.tooltips = [("file name", "@file_names"), ("topic summary", '@raw_topic_summaries')]

    #save(plot_lda, '20_news_tsne_lda_viz_{}_{}_{}_{}_{}_{}.html'.format(
    #    X_topics.shape[0], n_topics, threshold, n_iter, num_example, n_top_words))

    t2 = time.time()
    print ('\n>>> whole process done; took {} mins\n'.format((t2 - t0) / 60.))

    output_file("TSNE_OUTPUT.html", title="TTSNE OUTPUT")
    #show(plot_lda)

    
    
    script, div = components(plot_lda)
    
    
    return script, div