import pandas as pd
from db_connect import dbConnect
import sys
import os
currentdir = os.path.abspath(os.path.dirname(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

filename = parentdir+'/data/hw_param_retail_65.txt'
print(filename)
lst = []
with open(filename, "r") as f:
    content = f.readlines()
    #print(content)
for line in content:
    lst.append(line.strip())

breakpoint()
dbc = dbConnect()
tablename = 'HoltWinters_parameter_forecast_retail'
data = dbc.populate_analytical_db(df, tablename)

def create_staging_table(cursor) -> None:
    cursor.execute("""
        DROP TABLE IF EXISTS holtwinters_forecast_retail;
        CREATE UNLOGGED TABLE holtwinters_forecast_retail (
            id                  INTEGER,
            name                TEXT,
            tagline             TEXT,
            first_brewed        DATE,
            description         TEXT,
            image_url           TEXT,
            abv                 DECIMAL,
            ibu                 DECIMAL,
            target_fg           DECIMAL,
            target_og           DECIMAL,
            ebc                 DECIMAL,
            srm                 DECIMAL,
            ph                  DECIMAL,
            attenuation_level   DECIMAL,
            brewers_tips        TEXT,
            contributed_by      TEXT,
            volume              INTEGER
        );
    """)
