import PyPDF2
from nltk.tokenize import word_tokenize
from stopwords import stop_word_list
import docx
import nltk
import flask
from flask import session


def extract(filename):
    '''
    Extracts text from a given file and parses it.

    Parameters
    ----------
    filename : str
        Path of file in the filing system.

    Returns
    ----------
    text : str
        The raw text from the file. 
    tokens : list
        A list of the tokens from the raw text. 
    keywords : list
        A list of the tokens which are not a space, stop words, punctuation or special character.

    '''
    myid = session['myid']
    file_name_string = filename.split(".")
    file_format = file_name_string[-1]

    text = ""

    if file_format == "pdf":
        text = pdf_extract(filename)

    elif file_format == "docx":
        text = doc_extract(filename)

    elif file_format == "txt":
        text = text_extract(filename)

    else:
        added_extension = '_' + str(myid)
        real_filename = filename.replace(added_extension, '')
        raise Exception('Problem  with file:  ' + str(real_filename) +
                        '  This is not an allowed file type!')

    return simple_parse(text)

#########################################################################
# This method has been adapted from the following source:
# Link: https://medium.com/@rqaiserr/how-to-convert-pdfs-into-searchable-key-words-with-python-85aab86c544f
# Author: Rizwan Qaiser
# Date: 12/05/2017
#########################################################################


def simple_parse(text):
    '''
    Tokenizes the input text and returns the raw text, a list of tokens and a list of 'keywords'.
    Keywords are tokens which are not punctuation, stop words, spaces or special characters.

    Parameters
    ----------
    text : str
        Raw text to be tokenized. 

    Returns
    ----------
    text : str
        The raw text from the file. 
    tokens : list
        A list of the tokens from the raw text. 
    keywords : list
        A list of the tokens which are not a space, stop words, punctuation or special character.


    '''

    text = text.lower()
    stop_words = stop_word_list()
    #The word_tokenize() function will break our text phrases into #individual words
    tokens = word_tokenize(text)
    #we'll create a new list which contains punctuation we wish to clean
    punctuations = ['(', ')', ';', ':', '[', ']', ',', '.',
                    '-', '\"', '\'', '{', '}', ' - ']
    special_char = ['#', '<', '>',  '*', '+', ' - ', '~', '^', '"']
    #We initialize the stopwords variable which is a list of words like #"The", "I", "and", etc. that don't hold much value as keywords
    #We create a list comprehension which only returns a list of words #that are NOT IN stop_words and NOT IN punctuations.
    keywords = [
        word for word in tokens if not word in stop_words and not word in punctuations and not word in special_char]
    return text, tokens, keywords


#########################################################################
# This method has been adapted from the following source:
# Link: https://medium.com/@rqaiserr/how-to-convert-pdfs-into-searchable-key-words-with-python-85aab86c544f
# Author: Rizwan Qaiser
# Date: 12/05/2017
#########################################################################

def pdf_extract(filename):
    '''
    Extracts raw text from a pdf document. 
    Note, this will not work if it is not an annotated pdf, this does not provide OCR capabilities. 

    Parameters
    ----------
    filename : str
        Path to pdf file.

    Returns
    ----------
    text : str
        The raw text from the pdf file. 
    '''

    #open allows you to read the file
    pdfFileObj = open(filename, 'rb')
    #The pdfReader variable is a readable object that will be parsed
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    #discerning the number of pages will allow us to parse through all #the pages
    num_pages = pdfReader.numPages
    count = 0
    text = ""
    #The while loop will read each page

    while count < num_pages:
        pageObj = pdfReader.getPage(count)
        count += 1
        text += pageObj.extractText()
    #This if statement exists to check if the above library returned #words. It's done because PyPDF2 cannot read scanned files.
    if text != "":
        text = text
    #If the above returns as False, we run the OCR library textract to #convert scanned/image based PDF files into text
    else:
        raise Exception('The system cannot extract text from this file!')

    pdfFileObj.close()

    return text


def text_extract(filename):
    '''
    Extracts text from a txt file. 

    Parameters
    ----------
    filename : str
        Path to txt file.

    Returns
    ----------
    file_text : str
        The raw text from the txt file. 
    '''
    file = open(filename, 'r', errors='ignore')
    file_text = file.read()
    file.close()
    return file_text


#########################################################################
# This method has been adapted from the following source:
# Link: https://stackoverflow.com/questions/25228106/how-to-extract-text-from-an-existing-docx-file-using-python-docx
# Author: Chinmoy Panda
# Date: 08/03/2016
#########################################################################
def doc_extract(filename):
    '''
    Extracts text from a docx (word document) file. 

    Parameters
    ----------
    filename : str
        Path to docx file.

    Returns
    ----------
    The raw text from the docx file. 
    '''

    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)

    return '\n'.join(fullText)
