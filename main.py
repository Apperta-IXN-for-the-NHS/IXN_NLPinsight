from flask import render_template, Flask, flash, redirect, request, url_for, session, send_file
import uuid
from flask_session import Session
from config import Config
import flask
from flask_sqlalchemy import SQLAlchemy
import os
from shutil import copyfile
from cron_job import delete_daily
from database import db
import crython
import sys

app = Flask(__name__)
app.secret_key = '@\x10\x84\x9c\xef\x14!-s\x89\x89\x1e>\xc8\xa4\xcc\x8c\xe5\x9eZXN-Y'

#loads the app configurations from config.py
app.config.from_object(Config)

########################################################################## 
# This method has been adapted from the following source
# Link: https://stackoverflow.com/questions/22929839/circular-import-of-db-reference-using-flask-sqlalchemy-and-blueprints
# Author: S182
# Date: 1/05/2012
#########################################################################

#initialises the database
with app.app_context():
    db.init_app(app)

# crython is a cron job-like library that allows code to be executed at regular intervals.
crython.start()


#########################################################################
# This method has been adapted from the following source
# Link: https://github.com/ahawker/crython
# Author: Andrew Hawker
#########################################################################

# calls the delete_daily function from cron_job.py on a daily basis. This deletes all files that are older than 24hours.
@crython.job(expr='@daily')
def job():
    delete_daily()


print('>>>>>>>>>>>>>> APP INITIALIZED <<<<<<<<<<<<<<<<<<', file=sys.stdout)


from werkzeug.utils import secure_filename

from extractor import extract, simple_parse
from frequency_distribution import frequency_dist
from mywordcloud import build_word_cloud

import time
from lda_tsne_model2 import lda_tsne
from compressed_main import handle_compressed_file

import pickle
from mypyldavis import pyldavis_run
from flask_login import LoginManager, current_user, login_user, logout_user, login_required
from werkzeug.security import generate_password_hash, check_password_hash

from werkzeug.urls import url_parse

from azure.storage.blob import BlockBlobService, PublicAccess
import re


from models import User, Single_Upload, Group_Upload
from forms import LoginForm, UploadFileForm, inputText, inputTopicNumber, StopWordsForm, RegisterForm, EditAccountForm
import blob_upload
import zipfile
import shutil

# this defines where files are uploaded to
UPLOAD_FOLDER = '/uploads'

#this defines the set of allowed file extensions
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'zip', 'rar', 'docx'])

compressed_extensions = ['zip', 'rar']

#this sets the upload file limit to 50MB.
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024


login_settings = LoginManager(app)

login_settings.login_view = 'login_form'

#  updates the databse with the define data models from models.py
with app.app_context():
    db.create_all()

# creates the uploads directory and pickles directory. The former will be used to store uploaded files and their extracted versions.
# The later is used to store serialised python objects using the Python Pickle module (hence the name).
if not os.path.exists('uploads'):
    os.makedirs('uploads')
if not os.path.exists('pickles'):
    os.makedirs('pickles')


#########################################################################
# This method has been adapted from the following source
# Link: http://flask.pocoo.org/docs/1.0/patterns/fileuploads/
#########################################################################

# checks if the uploaded file has an allowed file extension.
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#########################################################################
# This method has been adapted from the following source
# Link: http://flask.pocoo.org/docs/1.0/patterns/fileuploads/
#########################################################################
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    '''
    This is the route function that governs the home page of the web app. 
    If there are not requests from the html file upload form the function will simply render the home.html template.
    If there is a POST request from the html file upload form, the uploaded file will be processed. 
    During the processing it is checked whether the upload file has an allowed extension (pdf,docx,txt for single files. rar,zip for compressed files).
    If the file is is a 'single file' (pdf, docx, txt), it will be stored in the 'uploads' folder and processed using the extractor.py module. 
    The processed text will then be used to build a wordcloud and a word-frequency graph.
    If the file is a 'compressed folder' (rar, zip), it will be decrompressed and the text extracted and processed usingt the compressed_main.py module. 
    The processed text will then be used to build the scatter graph of the clusters (using the lda_tsne_model2.py module) and the pyldavis visualisation (using the mypyldavis.py module).

    It must be noted that if less than 4 file are inside the compressed folder then the web app will trigger an error as the LDA model does not work with such little data.
    '''

    # This flask session object is used to determine whether the 'save' button will be visible or not.
    session['save'] = True

    # If the a user is not logged in, create a unique identifier for this specific user.
    # If the user is already logged in, the identifier has already been createrd.
    # the current_user.is_anonymous comes from the Flask-Login Flask extension.
    if current_user.is_anonymous:
        an_id = str(uuid.uuid4())
        myid = an_id[:8] + an_id[24:]
        session['myid'] = myid

    else:
        myid = session['myid']

    if request.method == 'POST':
        # check if the post request has the file part
        if 'document' not in request.files:
            print("file not in request.files")
            flash('No file part')
            return redirect(request.url)
        file = request.files['document']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)

            file_extension = filename.rsplit('.', 1)[1].lower()

            file_name_no_extension = filename.rsplit('.', 1)[0].lower()

            if len(file_name_no_extension) > 28:
                file_name_no_extension = file_name_no_extension[0:28]
                filename = file_name_no_extension + '.' + file_extension

            # defining the regular expresison of the allowed characters for the file name.
            regex = re.compile('[^a-zA-Z0-9-_]')

            #making sure that the file name contains only alpha-numeric and ('-', '_') characters. Any other character is deleted.
            file_name_no_extension = regex.sub('', file_name_no_extension)

            file_name_uuid = str(file_name_no_extension) + \
                '_' + str(myid) + '.' + file_extension
            file_name_uuid_no_extension = str(
                file_name_no_extension) + '_' + str(myid)

            file.save(os.path.join('uploads', file_name_uuid))

            session['single_file_path'] = os.path.join(
                'uploads', file_name_uuid)

            #checks if the uploaded file is a compressed format. This is pivotal in the program execution.
            if file_extension in compressed_extensions:

                # >>>>>>>>>>> This section of the code handles a compressed file (zip/rar)

                # creates the unique paths that are going to be used to store the serialised python objects (using the pickle module).
                # myid is the unique identifier that was previously created and is stored in session['myid']
                total_text_path = 'pickles/total_text_' + str(myid) + '.p'
                file_names_path = 'pickles/file_names_' + str(myid) + '.p'
                lda_model_path = "pickles/lda_model_" + str(myid) + '.p'
                lda_html_path = "pickles/lda_html_" + str(myid) + '.p'
                document_term_matrix_path = "pickles/document_term_matrix_" + \
                    str(myid) + '.p'
                cvectorizer_path = "pickles/cvectorizer_" + str(myid) + '.p'
                pyldavis_html_path = "pickles/pyldavis_html_" + \
                    str(myid) + '.p'

                # stores the various path variables in the flask session so that it can be used later.
                session['total_text_path'] = total_text_path
                session['file_names_path'] = file_names_path
                session['vectorizer_path'] = cvectorizer_path
                session['document_term_matrix_path'] = document_term_matrix_path
                session['lda_model_path'] = lda_model_path
                session['lda_html_path'] = lda_html_path
                session['pyldavis_html_path'] = pyldavis_html_path

                # handle_compressed_file decompresses the compressed file, extracts the text from each document within and parses/tokenizes/stems/removes stop words.
                # total_text is a list of strings where each element in the list represents the text from one document
                # totalvocab_stemmed is a list of stemmed words from the all the documents
                # totalvocab_tokenized is a list of text tokens from all the documents
                # file_names is a list of the file names from the compressed folder.
                total_text, totalvocab_stemmed, totalvocab_tokenized, file_names = handle_compressed_file(
                    (os.path.join('uploads', file_name_uuid)), filename)

                #once the compressed folder has been decrompressed and processed there is no need to keep it on the filing system so it is removed.
                os.remove(os.path.join('uploads', file_name_uuid))

                # LDA does not work with less than 4 files
                if len(file_names) < 4:
                    flash('At least 4 files in the compressed folder are required')
                    return redirect(url_for('upload_file'))

                # calls the lda_tsne method from lda_tsne_model2.py which peforms text vectorization, LDA clustering and then tSNE before converting the plot into html format.
                lda_html = lda_tsne(total_text, file_names)

                # Flask form for inputting a new number of topics parameter.
                topic_number_form = inputTopicNumber()

                #calls the pyldavis_run method from mypyldavis.py which produces the pyLDAvis visualization and converts it to html format.
                pyldavis_html = pyldavis_run(
                    lda_model_path, document_term_matrix_path, cvectorizer_path)

                # stores serialized versions of total_text, file_names, pyldavis_html and lda_html which will later be used.
                pickle.dump(total_text, open(total_text_path, "wb"))
                pickle.dump(file_names, open(file_names_path, "wb"))
                pickle.dump(pyldavis_html, open(pyldavis_html_path, "wb"))
                pickle.dump(lda_html, open(lda_html_path, "wb"))

                # Flask sessions object that dictates whether the 'Download' button is visible.
                session['download'] = True

                return render_template('bulk_analysis.html', title='Clustering analysis', lda_html=lda_html, number_form=topic_number_form, pyldavis_html=pyldavis_html)

            # >>>>>>>>>>> This section of the code handles a single file (docx/pdf/txt)

            #stores the name of the file name with and without the file extension and unique identifier in flask session objects.
            session['single_file_name_short_no_extension'] = file_name_no_extension
            session['single_file_name_uuid_long_no_extension'] = file_name_uuid_no_extension
            session['single_file_name_short_with_extension'] = filename
            session['single_file_name_long_with_extension'] = file_name_uuid

            # calls the extract method from extractor.py that extracts and processes the text from the document.
            text, tokens, keywords = extract(
                os.path.join('uploads', file_name_uuid))

            keywords_path = "pickles/keywords_" + str(myid) + '.p'

            pickle.dump(keywords, open(keywords_path, "wb"))

            # gets the pygal word-frequency distribution graph
            graph_data = frequency_dist(
                keywords, 26, ('Word frequency for file  with filename: ' + filename))

            # gets the wordcloud plot html
            wordcloud_html = build_word_cloud(text, 2000)

            wordcloud_html_path = "pickles/wordcloud_html_" + str(myid) + '.p'

            pickle.dump(wordcloud_html, open(wordcloud_html_path, "wb"))

            # gets the flask form that is used to input new stopwords.
            stop_words_form = StopWordsForm()

            session['title'] = 'Single file NLP analysis'

            return render_template('analysis_options.html', title='Single file NLP analysis', graph_data=graph_data, wordcloud_html=wordcloud_html,  stop_words_form=stop_words_form)

        else:
            flash('not an allowed file format')
            return redirect(url_for('upload_file'))
    else:
        uploadForm = UploadFileForm()
        inputTextForm = inputText()
        return render_template('home.html', title='Welcome', form=uploadForm, textform=inputTextForm)


@app.route('/submit', methods=['POST'])
def submit():
    '''
    This is the route function that deals with the text field submission from the homepage.
    It checks whether text has been entered. 
    If it has, the text will be parsed and tokenized.
    The word frequency graph and wordcloud will then be created similarly to the description above.

    '''

    myid = session['myid']

    if request.method == 'POST':

        #checks if text has been entered in the text area.
        if 'text' not in request.form:
            print("no text entered")
            flash('No text was entered')
            return redirect(request.url)

        text = request.form['text']

        #processes the text so it is tokenized and linguistic noise is removed.
        text, tokens, keywords = simple_parse(text)

        keywords_path = "pickles/keywords_" + str(myid) + '.p'

        pickle.dump(keywords, open(keywords_path, "wb"))

        # gets the pygal word-frequency object
        graph_data = frequency_dist(
            keywords, 26, ('Word frequency for input text'))

        # gets the wordcloud plot in html form
        wordcloud_html = build_word_cloud(text, 2000)

        wordcloud_html_path = "pickles/wordcloud_html_" + str(myid) + '.p'

        pickle.dump(wordcloud_html, open(wordcloud_html_path, "wb"))

        session['title'] = 'NLP analysis'

        stop_words_form = StopWordsForm()

        session['save'] = False

        return render_template('analysis_options.html', title='NLP analysis', graph_data=graph_data, wordcloud_html=wordcloud_html,  stop_words_form=stop_words_form)


#########################################################################
# This method has been adapted from the following source:
# Link: https://stackoverflow.com/questions/47368054/wtforms-test-whether-field-is-filled-out
# Author: Oluwafemi Sule
# Date: 23/11/17
#########################################################################
def is_filled(data):
    '''
    This method checks if an input field is empty or filled.
    '''
    if data == None:
        return False
    if data == '':
        return False
    if data == []:
        return False
    return True


@app.route('/submit_number_topics', methods=['POST'])
def submit_number_topics():
    '''
    This is the route function handles the submission of a new number of topics of top words for the LDA model. It re-runs the lda_tsne and pyldavis_run methods.
    '''

    myid = session['myid']

    # checks which of the two form fields contains data.
    if request.method == 'POST':
        if is_filled(request.form['number_topics']):
            number_topics = int(request.form['number_topics'])
            session['number_topics'] = str(number_topics)

        else:
            number_topics = int(session['number_topics'])

        if is_filled(request.form['number_topwords']):
            number_topwords = int(request.form['number_topwords'])
            session['number_topwords'] = str(number_topwords)

        else:
            number_topwords = int(session['number_topwords'])

        total_text_path = 'pickles/total_text_' + str(myid) + '.p'
        file_names_path = 'pickles/file_names_' + str(myid) + '.p'
        pyldavis_html_path = "pickles/pyldavis_html_" + str(myid) + '.p'

        # loads the serialized python objects for the total_text list and and the file_names list which are going to be used to re-run the lda_tsne method.
        total_text = pickle.load(open(total_text_path, "rb"))
        file_names = pickle.load(open(file_names_path, "rb"))

        lda_html_path = "pickles/lda_html_" + str(myid) + '.p'
        pyldavis_html_path = "pickles/pyldavis_html_" + str(myid) + '.p'

        # performs the lda clustering and tsne dimensionality reduction, resulting in scatter plot that is converted to html.
        lda_html = lda_tsne(total_text, file_names,
                            n_topics=number_topics, n_top_words=number_topwords)

        lda_model_path = "pickles/lda_model_" + str(myid) + '.p'
        document_term_matrix_path = "pickles/document_term_matrix_" + \
            str(myid) + '.p'
        cvectorizer_path = "pickles/cvectorizer_" + str(myid) + '.p'

        # if only the number of of top words is changing, there is no need to re-run the computationaly demanding pyldavis_run method. Instead the last output can simply be
        # loaded from the stored serilizaed object.
        # if however the number of topics has changed, the pyldavis_run method needs to be called as it also incorporates the number of topics within its calculations.
        if is_filled(request.form['number_topics']):
            pyldavis_html = pyldavis_run(
                lda_model_path, document_term_matrix_path, cvectorizer_path)
        else:
            pyldavis_html = pickle.load(open(pyldavis_html_path, "rb"))

        pickle.dump(pyldavis_html, open(pyldavis_html_path, "wb"))
        pickle.dump(lda_html, open(lda_html_path, "wb"))

        topic_number_form = inputTopicNumber()
        return render_template('bulk_analysis.html', title='Clustering analysis', lda_html=lda_html, number_form=topic_number_form, pyldavis_html=pyldavis_html)


@app.route('/submit_stop_words', methods=['POST'])
def submit_stop_words():
    '''
    This route function handles the submission of new stop words for the word-frequency distribution. 
    It processes the input text which should contain the new stop words delimited by commas. 
    It then re-reuns the frequency_dist method that produces the word-frequency distribution graph with the new stopwrods as one of the parameters. 
    '''

    myid = session['myid']

    if request.method == 'POST':
        if is_filled(request.form['stopwords']):
            new_stopwords = request.form['stopwords']
            new_stopwords = new_stopwords.replace(' ', '')
            new_stopwords = new_stopwords.split(",")

            keywords_path = "pickles/keywords_" + str(myid) + '.p'

            keywords = pickle.load(open(keywords_path, "rb"))
            title = session['title']
            graph_data = frequency_dist(keywords, 26, title, new_stopwords)

            wordcloud_html_path = "pickles/wordcloud_html_" + str(myid) + '.p'

            wordcloud_html = pickle.load(open(wordcloud_html_path, "rb"))

            stop_words_form = StopWordsForm()
            return render_template('analysis_options.html', title='Single file NLP analysis', graph_data=graph_data, stop_words_form=stop_words_form, wordcloud_html=wordcloud_html)


#########################################################################
# This method has been adapted from the following source:
# Link: https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-v-user-logins
# Author: Miguel Grinberg
# Date: 02/01/2018
#########################################################################
@login_settings.user_loader
def load_user(id):
    return User.query.get(int(id))


#########################################################################
# This method has been adapted from the following source:
# Link: https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-v-user-logins
# Author: Miguel Grinberg
# Date: 02/01/2018
#########################################################################
@app.route('/login_form', methods=['GET', 'POST'])
def login():
    '''
    This route function handles the login template and the login process. 
    It gets the form data, checks if the username and password are both correct and then calls the login_user() method 
    which comes from the Flask_login Flask extension.
    If the credentials are incorrect, an error message will be shown and the user will be redirected to the same page.
    When a user logs in, a unique identifier (myid) is created, this is stored in session['myid'] and is used to identify 
    any user files by adding the identifier to the file names.
    This helps the file deletion process as all that is required to delete all use files is to find all files 
    containing the particular identifier.
    '''

    if current_user.is_authenticated:
        return redirect(url_for('upload_file'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password')
            return redirect(url_for('login'))
        login_user(user)

        an_id = str(uuid.uuid4())
        myid = an_id[:8] + an_id[24:]
        session['myid'] = myid

        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            return redirect(url_for('upload_file'))
        return redirect(next_page)
    return render_template('login.html', title='Sign In', form=form)


#########################################################################
# This method has been adapted from the following source:
# Link: https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-v-user-logins
# Author: Miguel Grinberg
# Date: 02/01/2018
#########################################################################

@app.route('/logout')
@login_required
def logout():
    '''
    This route function logs the current user out using the 'logout_user()' method that is part of the Flask-Login Flask extension.
    The function can only be triggered when a user is logged in so there is no need to add in exeption handling or cases where a user is not logged in.
    '''
    logout_user()
    return redirect(url_for('upload_file'))


#########################################################################
# This method has been adapted from the following source:
# Link: https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-v-user-logins
# Author: Miguel Grinberg
# Date: 02/01/2018
#########################################################################

@app.route('/register_form', methods=['GET', 'POST'])
def register():
    '''
    This route function is called when a user registers new credentials on the web app.
    It handles the inputs and, hashes the password and adds the processed credentials to the SQLite database. 
    The form validation is handled in the forms.py module.
    '''

    if current_user.is_authenticated:
        return redirect(url_for('upload_file'))
    form = RegisterForm()
    if form.validate_on_submit():
        user = User(username=form.username.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('You are now a registered user!')
        return redirect(url_for('login'))
    return render_template('register.html', title='Submit', form=form)

#########################################################################
# This method has been adapted from the following source:
# Link: https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-v-user-logins
# Author: Miguel Grinberg
# Date: 02/01/2018
#########################################################################


@app.route('/my_account',  methods=['GET', 'POST'])
@login_required
def my_account():
    '''
    This route handles the 'my account' view and the change in user credentials.
    It loads the my_account.html template and populates the form fields with the existing credentials. 
    Note, for safety reasons the password is not shown.
    Note, the form validaiton for the change of credentials is handled by the forms.py module.
    '''

    def set_password(password):
        '''
        This method hashes the input password characters.
        '''
        return generate_password_hash(password)

    form = EditAccountForm()
    if form.validate_on_submit():
        current_user.username = form.username.data
        current_user.password_hash = set_password(form.new_password.data)
        db.session.commit()
        flash('Your changes have been saved.')
        return redirect(url_for('my_account'))

    elif request.method == 'GET':
        form.username.data = current_user.username
    return render_template('my_account.html', title='Edit Profile',
                           form=form)


@app.route('/save_group')
def save_group():
    '''
    This route function handles the process of saving the data of a compressed file input from a logged-in user.
    The process of saving the data involves saving the serialized pickle objects for the
    - total text list
    - file names list
    - vectorizer object (SciKitLearn CountVectorizer)
    - document-term matrix obtained from CountVectorizer
    - LDA model object
    - LDA model resulting html
    - pyldavis html object

    to Azure blob storage and saving the metadata to the SQLite database. 



    '''

    compressed_file_name_without_extension_uuid = session[
        'compressed_file_name_without_extension_uuid']
    compressed_file_name_with_extension = session['compressed_file_name']

    compressed_file_name_uuid = session['compressed_file_name_uuid']

    total_text_path = session['total_text_path']
    file_names_path = session['file_names_path']
    vectorizer_path = session['vectorizer_path']
    document_term_matrix_path = session['document_term_matrix_path']
    lda_model_path = session['lda_model_path']
    lda_html_path = session['lda_html_path']
    pyldavis_html_path = session['pyldavis_html_path']

    #########################################################################
    # This snippet has been adapted from the following source:
    # Link: https://stackoverflow.com/questions/22520932/python-remove-all-non-alphabet-chars-from-string
    # Author: limasxgoesto0
    # Date: 20/03/2014
    #########################################################################
    # deletes any non alphanumeric characters (and dash) in the compressed file name because 
    # Azure blob storage only allows for these characters in its container names.
    regex = re.compile('[^a-zA-Z0-9-]')
    compressed_file_name_without_extension_uuid = regex.sub(
        '', compressed_file_name_without_extension_uuid)

    #########################################################################
    # This snippet has been adapted from the following source:
    # Link: https://stackoverflow.com/questions/5998245/get-current-time-in-milliseconds-in-python
    # Author: Naftuli Kay
    # Date: 13/05/2011
    #########################################################################
    # converts the current time into one tenth of a milisecond
    millis = int(round(time.time() * 10000))

    container_name = compressed_file_name_without_extension_uuid + str(millis)

    #this method handles the Azure blob storage upload and the database update
    blob_upload.upload_group_file(compressed_file_name_with_extension, compressed_file_name_uuid, container_name, total_text_path,
                                  vectorizer_path, document_term_matrix_path, file_names_path, lda_model_path, lda_html_path, pyldavis_html_path, delete=False, update_db=True)

    lda_html = pickle.load(open(lda_html_path, "rb"))
    pyldavis_html = pickle.load(open(pyldavis_html_path, "rb"))

    topic_number_form = inputTopicNumber()

    flash('Your model has been saved')

    session['save'] = False

    return render_template('bulk_analysis.html', title='Clustering analysis', lda_html=lda_html, number_form=topic_number_form, pyldavis_html=pyldavis_html)


@app.route('/save_single')
def save_single():
    '''
    This route function handles the process of saving the data of a single file input from a logged-in user.
    The process of saving the data involves saving the actual uploaded file into Azure blob storage and saving the metadata to the SQLite database. 
    Since single files are usually small in terms of memory and the processing of these is quick, there is little advantage from serializing the resulting 
    Python objects and saving these instead of the original file. 

    '''

    myid = session['myid']

    single_file_name_short_no_extension = session['single_file_name_short_no_extension']
    single_file_name_uuid_long_no_extension = session['single_file_name_uuid_long_no_extension']
    single_file_name_short_with_extension = session['single_file_name_short_with_extension']
    single_file_name_long_with_extension = session['single_file_name_long_with_extension']

    #########################################################################
    # This snippet has been adapted from the following source:
    # Link: https://stackoverflow.com/questions/22520932/python-remove-all-non-alphabet-chars-from-string
    # Author: limasxgoesto0
    # Date: 20/03/2014
    #########################################################################
    # deletes any non alphanumeric characters (and dash) in the compressed file name because Azure blob storage only allows for these characters in its container names.
    regex = re.compile('[^a-zA-Z0-9-]')
    single_file_name_uuid_long_no_extension = regex.sub(
        '', single_file_name_uuid_long_no_extension)

    single_file_path = os.path.join(
        'uploads', single_file_name_long_with_extension)

    #########################################################################
    # This snippet has been adapted from the following source:
    # Link: https://stackoverflow.com/questions/5998245/get-current-time-in-milliseconds-in-python
    # Author: Naftuli Kay
    # Date: 13/05/2011
    #########################################################################
    # converts the current time into one tenth of a milisecond
    millis = int(round(time.time() * 10000))
    container_name = single_file_name_uuid_long_no_extension + str(millis)

    #this method handles the Azure blob storage upload and the database update
    blob_upload.upload_single_file(single_file_name_short_with_extension, single_file_path,
                                   container_name, single_file_name_long_with_extension,   delete=False, update_db=True)

    # the next few lines reload the template after the form submission.
    keywords_path = "pickles/keywords_" + str(myid) + '.p'

    keywords = pickle.load(open(keywords_path, "rb"))
    title = session['title']
    graph_data = frequency_dist(keywords, 26, title)

    wordcloud_html_path = "pickles/wordcloud_html_" + str(myid) + '.p'

    wordcloud_html = pickle.load(open(wordcloud_html_path, "rb"))

    stop_words_form = StopWordsForm()
    flash('Your model has been saved')

    # hides the 'save' button as there is no need to save the same model twice.
    session['save'] = False

    return render_template('analysis_options.html', title='Single file NLP analysis', graph_data=graph_data, stop_words_form=stop_words_form, wordcloud_html=wordcloud_html)


def delete_files(myid):
    '''
    Method that deletes any file that contains 'myid' in its name in the used directories. 
    The three used directories are 'uploads', 'uploads/extracted' and 'pickles'.
    '''
    uploads_path = 'uploads'
    extracted_path = 'uploads/extracted'
    pickles_path = 'pickles'
    for i in os.listdir(uploads_path):
        if os.path.isfile(os.path.join(uploads_path, i)) and str(myid) in i:
            print('>>>> Removing ' + str(os.path.join(uploads_path, i)))
            os.remove(os.path.join(uploads_path, i))

    for i in os.listdir(extracted_path):
        if os.path.isfile(os.path.join(extracted_path, i)) and str(myid) in i:
            print('>>>> Removing ' + str(os.path.join(extracted_path, i)))
            os.remove(os.path.join(extracted_path, i))

    for i in os.listdir(pickles_path):
        if os.path.isfile(os.path.join(pickles_path, i)) and str(myid) in i:
            print('>>>> Removing ' + str(os.path.join(pickles_path, i)))
            os.remove(os.path.join(pickles_path, i))


@app.route('/history')
@login_required
def history():
    '''
    This route function renders the history.html template and loads the previously saved files by the user.

    '''

    current_username = current_user.username
    user = User.query.filter_by(username=current_username).first()
    user_id = user.id
    single_files = Single_Upload.query.filter_by(user_id=user_id)
    group_files = Group_Upload.query.filter_by(user_id=user_id)

    return render_template('history.html', title='History', single_files=single_files, group_files=group_files)


@app.route('/history_single/<container_name>')
@login_required
def history_single(container_name):
    '''
    This route function handles the rendering of a previously saved single file. 
    It is called when a user clicks on a specific element in the single file history table. 
    This querries the the database for the details of the Azure blob storage container. 
    It then downloads the specific file from Azure into the save_path directory. 
    Once this is done, it calls the display_history_single method to process the downloaded file 
    and display the wordcloud and word-frequency graph.
    '''

    myid = session['myid']

    #query  the database for the container metadata.
    single_file = Single_Upload.query.filter_by(
        container_name=container_name).first()

    block_blob_service = BlockBlobService(
        account_name='mednlpstorage', account_key='v+IgtNIIRhZjqMZx+e886rhJMVAhIUoUfG252SVIftBCyx8bG+NE0apP20xakOsMRQfNZFbUggUUULN2JER8lg==')

    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    file_name_short_without_extension = single_file.file_name_short_with_extension.rsplit('.', 1)[
        0].lower()

    file_extension = single_file.file_name_short_with_extension.rsplit('.', 1)[
        1].lower()

    mypath = file_name_short_without_extension + \
        '_' + str(myid) + '.' + file_extension

    save_path = os.path.join('uploads', mypath)

    #queries the Azure blob storage for the files inside the specfic container.
    block_blob_service.get_blob_to_path(
        single_file.container_name, single_file.file_name_long_with_extension, save_path)

    return redirect(url_for('display_history_single',  save_path=save_path, file_name_short_with_extension=single_file.file_name_short_with_extension))


@app.route('/history_group/<container_name>')
@login_required
def history_group(container_name):
    '''
    This route function handles the rendering of a previously saved compressed file.
    It is called when a user clicks on a specific element in the compressed file history table. 
    This querries the the database for the details of the Azure blob storage container. 
    It then downloads the specific files from Azure to the specific paths.
    Once this is done, it calls the display_history_group method to process the
    downloaded files and display the scatter graph and the pyldavis visualisation.
    '''

    myid = session['myid']

    group_file = Group_Upload.query.filter_by(
        container_name=container_name).first()
    block_blob_service = BlockBlobService(
        account_name='mednlpstorage', account_key='v+IgtNIIRhZjqMZx+e886rhJMVAhIUoUfG252SVIftBCyx8bG+NE0apP20xakOsMRQfNZFbUggUUULN2JER8lg==')

    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    #directory_path = os.path.join('uploads', group_file.container_name)

    total_text_path = 'pickles/total_text_' + str(myid) + '.p'
    file_names_path = 'pickles/file_names_' + str(myid) + '.p'
    pyldavis_html_path = "pickles/pyldavis_html_" + str(myid) + '.p'
    lda_html_path = "pickles/lda_html_" + str(myid) + '.p'
    lda_model_path = "pickles/lda_model_" + str(myid) + '.p'
    document_term_matrix_path = "pickles/document_term_matrix_" + \
        str(myid) + '.p'
    cvectorizer_path = "pickles/cvectorizer_" + str(myid) + '.p'

    block_blob_service.get_blob_to_path(
        group_file.container_name, 'total_text.p', total_text_path)
    block_blob_service.get_blob_to_path(
        group_file.container_name, 'vectorizer.p', cvectorizer_path)
    block_blob_service.get_blob_to_path(
        group_file.container_name, 'dtm.p',  document_term_matrix_path)
    block_blob_service.get_blob_to_path(
        group_file.container_name, 'file-names.p',  file_names_path)
    block_blob_service.get_blob_to_path(
        group_file.container_name, 'lda-model.p',  lda_model_path)
    block_blob_service.get_blob_to_path(
        group_file.container_name, 'lda-html.p', lda_html_path)
    block_blob_service.get_blob_to_path(
        group_file.container_name, 'pyldavis-html.p',   pyldavis_html_path)

    return redirect(url_for('display_history_group'))


@app.route('/display_history_single')
@login_required
def display_history_single():
    '''
    This route function handles the loaded single file from Azure blob storage,
    processes it and displays the wordcloud and the word-frequency distribution graph.
    This has a lot of similarity how an uploaded single file is handled.
    '''

    myid = session['myid']

    save_path = request.args['save_path']

    file_name_short_with_extension = request.args['file_name_short_with_extension']

    text, tokens, keywords = extract(save_path)

    keywords_path = "pickles/keywords_" + str(myid) + '.p'

    pickle.dump(keywords, open(keywords_path, "wb"))

    graph_data = frequency_dist(
        keywords, 26, ('Word frequency for file  with filename: ' + file_name_short_with_extension))

    wordcloud_html = build_word_cloud(text, 2000)

    wordcloud_html_path = "pickles/wordcloud_html_" + str(myid) + '.p'

    pickle.dump(wordcloud_html, open(wordcloud_html_path, "wb"))

    session['save'] = False

    stop_words_form = StopWordsForm()

    return render_template('analysis_options.html', title='Single file NLP analysis', graph_data=graph_data, stop_words_form=stop_words_form, wordcloud_html=wordcloud_html)


@app.route('/display_history_group')
@login_required
def display_history_group():
    '''
    This route function handles the loaded single file from Azure blob storage, 
    processes it and displays the wordcloud and the word-frequency distribution graph.
    This has a lot of similarity how an uploaded single file is handled.
    '''

    myid = session['myid']

    pyldavis_html_path = "pickles/pyldavis_html_" + str(myid) + '.p'
    lda_html_path = "pickles/lda_html_" + str(myid) + '.p'

    lda_html = pickle.load(open(lda_html_path, "rb"))
    pyldavis_html = pickle.load(open(pyldavis_html_path, "rb"))

    topic_number_form = inputTopicNumber()

    # there is no need to save a model that has already been saved before
    session['save'] = False
    # the downloaded model has no access to the original uploaded files and therefore cannot produce a folder containing these files.
    session['download'] = False

    return render_template('bulk_analysis.html', title='Clustering analysis', lda_html=lda_html, number_form=topic_number_form, pyldavis_html=pyldavis_html)


@app.route('/delete_all')
@login_required
def delete_all():
    '''
    This route function deletes all files that have been saved by the user. 
    They are removed from Azure blobd storage and any metadata is removed from the SQLite database.

    '''

    myid = session['myid']
    blob_upload.delete_all()
    delete_files(str(myid))

    flash('All your files have been deleted')

    current_username = current_user.username
    user = User.query.filter_by(username=current_username).first()
    user_id = user.id
    single_files = Single_Upload.query.filter_by(user_id=user_id)
    group_files = Group_Upload.query.filter_by(user_id=user_id)

    return render_template('history.html', title='History', single_files=single_files, group_files=group_files)


@app.route('/download_docs')
def download_docs():
    '''
    This route function allows the user (logged in or not) to download the uploaded compressed folder (zip or rar) with the files classified 
    following the clustering pattern. Each cluster will be contained within a separate folder.
    The file is compressed (.zip format) before the download.

    The 'lda_keys' list maps each inidividual file to the its predicted topic(cluster). 
    The 'raw_topic_summaries' list maps each inidividual file to the topic summary using 
    the top words for each specific topic.

    '''

    myid = session['myid']

    raw_topic_summaries_path = "pickles/raw_topic_summaries" + str(myid) + '.p'
    lda_keys_path = "pickles/lda_keys_path" + str(myid) + '.p'

    raw_topic_summaries = pickle.load(open(raw_topic_summaries_path, "rb"))
    lda_keys = pickle.load(open(lda_keys_path, "rb"))

    original_directory = session['compressed_file_name_without_extension_uuid']
    print(original_directory)

    if not os.path.exists('download_file'):
        os.makedirs('download_file')

    for x in raw_topic_summaries:
        x = x.replace(' ', '-')
        path = os.path.join('download_file', str(x))
        if not os.path.exists(path):
            os.makedirs(path)

    count = 0
    for filename in os.listdir(os.path.join('uploads/extracted', str(original_directory))):
        src_path = os.path.join('uploads/extracted/' +
                                str(original_directory), filename)
        y = raw_topic_summaries[count]
        y = y.replace(' ', '-')
        dst = os.path.join('download_file', y)
        dst = os.path.join(dst, filename)
        count = count + 1
        copyfile(src_path, dst)

    output_filename = 'classified_output' + str(myid)

    zip_path = os.path.join('uploads',  output_filename)

    shutil.make_archive(zip_path, 'zip', 'download_file')

    # the download_file directory is deleted as it is only a temporary directory 
    # that was used for the processing of the files.
    shutil.rmtree('download_file')

    return send_file(zip_path + '.zip', as_attachment=True, attachment_filename='classified_output.zip')


@app.route('/about')
def about():
    '''
    Renders the 'about.html' template.
    '''
    return render_template('about.html')


if __name__ == '__main__':
    #threaded here allows for multiple users to access the web app at the same time.
    app.run(host='0.0.0.0', port=80, threaded=True)
