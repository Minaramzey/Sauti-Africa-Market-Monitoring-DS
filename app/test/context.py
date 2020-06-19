import os
import sys

currentdir = os.path.abspath(os.path.dirname(__file__))
parentdir = os.path.dirname(currentdir)
#siblingdir = os.path.join(parentdir, 'code') # no need 

sys.path.insert(0, currentdir)
sys.path.insert(0, parentdir)
#sys.path.insert(0,siblingdir)

# print(currentdir)
# print(parentdir)
#print(siblingdir)
