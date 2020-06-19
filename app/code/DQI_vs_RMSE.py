
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_process import DataCleaning, DataQualityCheck
from db_connect import dbConnect
from classic_forecast import grid_search_one_step, ModelConfig
import time 
## test time series
sale_type = 'retail'

## get quality info for screening time series
db_c = dbConnect()
qc_tablename = 'qc_'+sale_type
qc_table = db_c.read_analytical_db(qc_tablename)
# return only great data quality
qc = qc_table[qc_table['DQI_cat'] == 'great']

hw_tablename = 'hw_config_'+sale_type 
hw = db_c.read_analytical_db(hw_tablename)

new_table = qc.merge(hw, how='left', on='qc_id')

tablename='DQI_RMSE_'+'retail'
db_c.populate_analytical_db(new_table, tablename)