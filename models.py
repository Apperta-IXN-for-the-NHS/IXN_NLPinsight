
from database import db
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from sqlalchemy import Column, Integer, DateTime
from sqlalchemy.sql import func

#########################################################################
# This code has been adapted from the following source:
# Link: https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-v-user-logins
# Author: Miguel Grinberg
# Date: 02/01/2018
#########################################################################


class User(UserMixin, db.Model):
    '''
    Data model that represents the user in the database.
    '''
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    password_hash = db.Column(db.String(128))

    def __repr__(self):
        return '<User {}>'.format(self.username)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Single_Upload(db.Model):
    '''
    Data model that represents a single file upload in the database.
    '''
    container_name = db.Column(db.String(256), primary_key=True)
    file_name_short_with_extension = db.Column(db.String(256))
    file_name_long_with_extension = db.Column(db.String(256))
    date_created = db.Column(db.DateTime, default=func.now())
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

    def __repr__(self):
        return '<File Name {}>'.format(self.file_name_short_with_extension)


class Group_Upload(db.Model):
    '''
    Data model that represents a compressed file upload in the database.
    '''
    container_name = db.Column(db.String(256), primary_key=True)
    compressed_file_name_short_with_extension = db.Column(db.String(256))
    date_created = db.Column(db.DateTime, default=func.now())
    compressed_file_name_long_with_extension = db.Column(db.String(256))
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

    def __repr__(self):
        return '<File Name {}>'.format(self.compressed_file_name_short_with_extension)
