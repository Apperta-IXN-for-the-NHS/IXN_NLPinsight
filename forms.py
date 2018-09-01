from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from werkzeug.utils import secure_filename
from wtforms import StringField, PasswordField, BooleanField, SubmitField, TextAreaField, IntegerField, FloatField
from wtforms.validators import DataRequired
from wtforms.validators import ValidationError, DataRequired, EqualTo, Length
from models import User
from flask_login import current_user
from werkzeug.security import generate_password_hash, check_password_hash

#########################################################################
# This code has been adapted from the following source:
# Link: https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-v-user-logins
# Author: Miguel Grinberg
# Date: 02/01/2018
#########################################################################


class LoginForm(FlaskForm):
    '''
    Form for user login.
    '''
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Submit')

# password2 is to re-enter the password for confirmaiton. The validator that checks if it matches the first password is  'EqualTo('password')' .


class RegisterForm(FlaskForm):
    '''
    Form for new user registration.
    '''
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    password2 = PasswordField('Retype Password', validators=[
                              DataRequired(),  EqualTo('password')])
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError(
                'Username already in use. Please use a different username.')


class StopWordsForm(FlaskForm):
    '''
    Form for inputing new sotp words for the word-frequency distribution graph.

    '''
    stopwords = StringField('new_stop_words', validators=[
                            DataRequired()], render_kw={"placeholder": "Stopwords"})
    submit = SubmitField('Submit')


class UploadFileForm(FlaskForm):
    '''
    Form for uploading file on the home.html template.
    '''
    document = FileField('document', validators=[FileRequired])
    submit = SubmitField('Upload')


class inputText(FlaskForm):
    '''
    Form for the input text field on the home.html template.
    '''
    text = TextAreaField('input_text', validators=[DataRequired])
    submit = SubmitField('Submit')


class inputTopicNumber(FlaskForm):
    '''
    Form for the input of new number of topics and new number of top words for the LDA clustering.

    '''
    number_topics = IntegerField('number_topics', render_kw={
                                 "placeholder": "New number of topics"})
    number_topwords = IntegerField('number_topwords', render_kw={
                                   "placeholder": "New number of top words"})
    submit = SubmitField('Submit')


class EditAccountForm(FlaskForm):
    '''
    Form for displaying and editing user account information.
    '''
    username = StringField('Username', validators=[DataRequired()])
    current_password = PasswordField(
        'Current Password', validators=[DataRequired()])
    new_password = PasswordField('New Password', validators=[DataRequired()])
    new_password2 = PasswordField('Retype New Password', validators=[
                                  DataRequired(),  EqualTo('new_password')])
    submit = SubmitField('Submit')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if (user is not None) and (username.data != current_user.username):
            raise ValidationError(
                'Username already in use. Please use a different username.')

    def validate_current_password(self,  current_password):
         if not current_user.check_password(current_password.data):
              raise ValidationError('Incorrect current password')
