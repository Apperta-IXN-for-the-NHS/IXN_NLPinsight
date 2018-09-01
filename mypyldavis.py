#########################################################################
# This code has been adapted from the following two sources:
# Link: http://nbviewer.jupyter.org/github/bmabey/hacker_news_topic_modelling/blob/master/HN%20Topic%20Model%20Talk.ipynb#topic=1&lambda=1&term=
#Link: http://nbviewer.jupyter.org/github/bmabey/pyLDAvis/blob/master/notebooks/pyLDAvis_overview.ipynb
# Author: Ben Mabey
#########################################################################

import os
import pyLDAvis
from pyLDAvis.sklearn import prepare
import time
import pickle
import sys


def pyldavis_run(lda_model_path, document_term_matrix_path, vectorizer_path):
    '''
    Computes the pyLDAvis visualisation of the LDA model.

    Parameters
    ----------
    lda_model_ath : str
        Path of the  pickle object (serialised python object) of the LDA model. This is created in the lda_tsne_model2.py module.
    document_term_matrix_path : str
        Path of the  pickle object (serialised python object) of the document-term matrix which is created using the CountVectorizer in the lda_tsne_model2.py module.
    vectorizer_path : str 
        Path of the  pickle object (serialised python object) of the vectorizer used to create the document-term matrix.This is usually the CountVectorizer in the lda_tsne_model2.py module.

    Returns
    ----------
    Embedded html pyldavis visulisation of the LDA model.
    '''

    t0 = time.time()

    # loading the pickle objects from the paths parameters.
    lda_model = pickle.load(open(lda_model_path, "rb"))
    document_term_matrix = pickle.load(open(document_term_matrix_path, "rb"))
    cvectorizer = pickle.load(open(vectorizer_path, "rb"))

    #prepares the pyldavis visualisation. There is a choice of dimensionality reduction methods here, TSNE is chosen as it is consistent
    #with the previous analysis in the lda_tsne_model2.py module and has shown to yield better results than other available methods.
    prepared_data = prepare(lda_model, document_term_matrix,
                            cvectorizer, mds='tsne', plot_opts={'xlab': '', 'ylab': ''})

    html = pyLDAvis.prepared_data_to_html(prepared_data)

    t1 = time.time()
    print("time for pyldavis: " + str(t1-t0), file=sys.stdout)

    return html
