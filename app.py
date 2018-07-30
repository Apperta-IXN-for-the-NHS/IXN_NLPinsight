from flask import render_template, Flask, flash, redirect, request, url_for, send_from_directory, make_response
from config import Config
from forms import LoginForm, UploadFileForm, inputText
from werkzeug.utils import secure_filename
import os
from extractor import extract, simple_parse
from frequency_distribution import frequency_dist
from mywordcloud import build_word_cloud
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import urllib.parse
import time
from lda_tsne_model import lda_tsne
from compressed_main import handle_compressed_file




app = Flask(__name__)

UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'zip','rar', 'docx'])

compressed_extensions = ['zip', 'rar']


app.config.from_object(Config)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024



@app.route('/')
def hello():
    uploadForm = UploadFileForm()
    inputTextForm = inputText()
    return render_template('home.html',title = 'Welcome', form = uploadForm, textform = inputTextForm)




@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    return render_template('login.html', title='Submit', form=form)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads',
                               filename)



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/', methods=['GET', 'POST'])
def upload_file():
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

            file_extension =  filename.rsplit('.', 1)[1].lower()

            if not os.path.exists('uploads'):
                os.makedirs('uploads')

            file.save(os.path.join('uploads', filename))

            if file_extension in compressed_extensions:
                total_text, totalvocab_stemmed, totalvocab_tokenized, file_names = handle_compressed_file((os.path.join('uploads', filename)), filename)
                if len(file_names)<4:
                    flash('At least 4 files in the compressed folder are required')
                    return redirect(url_for('upload_file'))
                script, div = lda_tsne(total_text, file_names)
                return render_template('bulk_analysis.html', title = 'Clustering analysis', script = script, div= div)
            
            text, tokens, keywords = extract(os.path.join('uploads', filename))
            graph_data = frequency_dist(keywords, 26, ('Word frequency for file  with filename: ' + filename))

            current_time = str(time.time())
            current_time = current_time.replace(".", "")
            
            myextension = str(current_time)


            build_word_cloud(text, 2000, myextension)

            myfilename = '/static/mycloud' + myextension + '.png'

          
            
            return render_template('analysis_options.html', title='Single file NLP analysis', graph_data = graph_data, myfilename = myfilename)
        
   
        else:
            flash('not an allowed file format')
            return redirect(url_for('upload_file'))




@app.route('/submit', methods=['POST'])
def submit():
     if request.method == 'POST':
        # check if the post request has the file part
        if 'text' not in request.form:
            print("no text entered")
            flash('No text was entered')
            return redirect(request.url)

        text = request.form['text']
   
        text, tokens, keywords = simple_parse(text)
         
        graph_data = frequency_dist(keywords, 26, ('Word frequency for input text'))

        current_time = str(time.time())
        current_time = current_time.replace(".", "")
        
        myextension = str(current_time)


        build_word_cloud(text, 2000, myextension)

        myfilename = '/static/mycloud' + myextension + '.png'
    
        return render_template('analysis_options.html', title='NLP analysis', graph_data = graph_data, myfilename = myfilename)






if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)