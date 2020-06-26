import pandas as pd
from db_connect import dbConnect
import sys
import re
import os
currentdir = os.path.abspath(os.path.dirname(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
'''
dbc = dbConnect()
connection = dbc.connect_analytical_db()
cursor = connection.cursor()

cursor.execute('''
''')

'''
filename = parentdir+'/data/hw_param_wholesale_65.txt'
print(filename)
lst = []
with open(filename, "r") as f:
    content = f.readlines()
    #print(content)


lst = []
for line in content:
    row = line.strip()
    x = re.split(r',\s*(?![^()]*\))', row)
    lst.append([x[0:4], x[4], x[5], x[6], x[7:13], x[13], x[14], x[15:]])

print(lst)

df = pd.DataFrame(data=lst, index=None, columns=['ts_name', 'currency', 'last_date', 'last_observation', 'cfg_selected',
"rmspe_percent", 'month_avg_forecast', 'month_forecast'])


dbc = dbConnect()
tablename = 'HoltWinters_parameter_forecast_wholesale'
data = dbc.populate_analytical_db(df, tablename)


