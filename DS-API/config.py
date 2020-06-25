import os

from dotenv import load_dotenv

load_dotenv()

class Config():
    ''' Base configuration reading from .env file'''

    DEBUG = os.environ.get('DEBUG')
    SECRET_KEY = os.environ.get('SECRET_KEY')
    JSON_SORT_KEYS = False