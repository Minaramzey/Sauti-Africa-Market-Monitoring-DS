import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print(sys.path)

#from app import flask_app
print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))

#import  app.modules.test.test_sequence # this will set package module names https://napuzba.com/a/import-error-relative-no-parent/p3