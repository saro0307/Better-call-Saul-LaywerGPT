from . import db
# from flask_login import UserMixin


class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    bot = db.Column(db.String(500))
    user = db.Column(db.String(50))