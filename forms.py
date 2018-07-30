from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from werkzeug.utils import secure_filename
from wtforms import StringField, PasswordField, BooleanField, SubmitField, TextAreaField
from wtforms.validators import DataRequired


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Submit')



class UploadFileForm(FlaskForm):
    document = FileField('document', validators = [FileRequired])
    submit = SubmitField('Upload')


class inputText(FlaskForm):
    text = TextAreaField('input_text', validators = [DataRequired])
    submit = SubmitField('Submit')