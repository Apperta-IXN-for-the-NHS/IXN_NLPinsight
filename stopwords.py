from spacy.lang.en.stop_words import STOP_WORDS
import nltk

#STOP_WORDS.add("your_additional_stop_word_here")


def stop_word_list():
    '''
    Computes a list of stopwords. The NLTK and Spacy libraries both have pre-defined lists of stopwords. 
    These are combined to make a larger set of possible english stopwords. 

    The method returns a list of strings where each string represents a stop word.

    '''

    nltk_stopwords = nltk.corpus.stopwords.words('english')

    for word in STOP_WORDS:
        if word not in nltk_stopwords:
            nltk_stopwords.append(word)

    #The test data had a disproportionate frequency of the words below. These were added as stopwords.
    #nltk_stopwords.append('Description')
    #nltk_stopwords.append('Diagnosis')
    #nltk_stopwords.append('description')
    #nltk_stopwords.append('diagnosis')

    return nltk_stopwords
