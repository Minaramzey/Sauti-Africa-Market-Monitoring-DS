"""grid-search for all database series using holter-winter method"""
import context
import time
# from math import sqrt, floor
# import numpy as np
# import pandas as pd

from code.db_connect import dbConnect
from code.data_process_v2 import DataCleaning, DataQualityCheck, TimeSeries
from code.classic_forecast_v2 import ForecastModels, ModelConfig, grid_search_slide_window, walk_forward_validation_slide_window, train_test_split, score_model, measure_rmspe

## test time series
sale_type = 'retail'

## get source id map
db_c = dbConnect()
source_tablename = 'sources'
source_table = db_c.read_analytical_db(source_tablename)

## get quality data for grid search
qc_tablename = 'qc_' + sale_type + '_observed_price'
qc_table = db_c.read_analytical_db(qc_tablename)
# top quality time series table
qc = qc_table[qc_table['DQI_cat'] == 'great'].sort_values(by ='DQI', ascending = False)
print(qc.head())
breakpoint()

"""model config"""
mc = ModelConfig()
# config list for one time series
cfg_list = mc.exp_smoothing_configs(seasonal=[6, 12])
print(f'A total of {len(cfg_list)} configurations.')

for idx in range(len(qc)):
    # get right parameters for ts object
    market_id = qc.iloc[idx,1]
    product_name = qc.iloc[idx,2]
    source_id = qc.iloc[idx,3]
    breakpoint()
    market_name = market_id.split(' : ')[0]
    x = source_table.loc[source_table['id'] == source_id, 'source_name']
    source_name = x[0]
    print(f'Market: {market_name},  product: {product_name}, source: {source_name}')
    # get time series data from stakeholder db
    ts = TimeSeries()
    # extract and clean time series
    
    # extract names to match stakeholder table cols
    ts.extract_data(market=market_name, product=product_name, source=source_name, sale_type=sale_type)
    sale = ts.data
    ts_name = ts.name
    ts_currency = ts.currency
    
    # interpolate for missing values    sale = sale.interpolate(method='nearest')

    # split sale into val and data
    n_val = 30
    val = sale[-n_val:]
    data = sale[: -n_val]
    # split data into train and test, 40% for train start length
    n_test = round(len(data)*0.6) 
    window_length = 30
    slide_distance = 7
    train, test = train_test_split(data, n_test)
    print(f'Time series data length: train={len(train)}, test={len(test)}, val={len(val)}')

    #test grid search
    start_time = time.time() 
    scores = grid_search_slide_window(data, n_test, window_length, slide_distance, cfg_list, parallel=True)
    elapsed_time = time.time() - start_time
    print(f"elapsed time is {elapsed_time} sec.")
    
    # top score
    if scores != []:
        error, cfg_selected = scores[1]
        results = [ts_name, ts_currency, cfg_selected, error]
        # convert to str
        results = [str(item) for item in results]

        #save tuple list to txt file
        filename = 'hw_param_'+sale_type+'.txt'
        f = open(filename, 'a+')
        # a for append
        line = ', '.join(x for x in results)
        f.write(line + '\n')
        f.close()
    else:
        break        