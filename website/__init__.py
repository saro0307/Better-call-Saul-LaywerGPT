from flask import Flask, Blueprint
from flask_sqlalchemy import SQLAlchemy

DB_NAME = "database.db"
db = SQLAlchemy()


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = "abc123"
    app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{DB_NAME}"
    db.init_app(app)
    from .views import views
    app.register_blueprint(views, url_prefix='/')
    from .models import History
    with app.app_context():
        db.create_all()
        print("Database created!")
    return app
