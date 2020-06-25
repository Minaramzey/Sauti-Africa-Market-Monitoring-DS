import urllib.parse as up

from flask import Flask
from flask_cors import CORS

def create_app():

    app = Flask(__name__, instance_relative_config=False) # The configuration is comming from other file.
    app.config.from_object('config.Config')
    CORS(app)

    with app.app_context():

        # Import routes

        from . import routes

        return app