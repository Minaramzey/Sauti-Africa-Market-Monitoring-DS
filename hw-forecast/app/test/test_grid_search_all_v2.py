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
SALE_TYPE = 'retail'

## get source id map
db_c = dbConnect()
source_tablename = 'sources'
source_table = db_c.read_analytical_db(source_tablename)

## get quality data for grid search
qc_tablename = 'qc_' + SALE_TYPE + '_observed_price'
qc_table = db_c.read_analytical_db(qc_tablename)
# top quality time series table
qc = qc_table[qc_table['DQI_cat'] == 'great'].sort_values(by ='DQI', ascending = False)
print(qc.head())

"""model config"""
mc = ModelConfig()
# config list for one time series
cfg_list = mc.exp_smoothing_configs(seasonal=[6, 12])
print(f'A total of {len(cfg_list)} configurations.')

for idx in range(len(qc)):
    # get right parameters for ts object
    MARKET = qc.iloc[idx,1]
    PRODUCT = qc.iloc[idx,2]
    SOURCE = qc.iloc[idx,3]
    
    ts.extract_data(MARKET, PRODUCT, SOURCE, SALE_TYPE)
    sale = ts.data
    ts_name = ts.name
    ts_currency = ts.currency
    # interpolate for missing values
    sale = sale.interpolate(method='nearest')
    
    """prepare data"""
    # split sale into val and data
    n_val = 30
    val = sale[-n_val:]
    data = sale[: -n_val]
    # split data into train and test
    n_test = round(len(data)*0.4)
    window_length = 30
    slide_distance = 14
    train, test = train_test_split(data, n_test)
    print(f'Data length: train={len(train)}, test={len(test)}, val={len(val)}')

    start_time = time.time()
    scores = grid_search_slide_window(
        data, n_test, window_length, slide_distance, cfg_list, parallel=True)
    elapsed_time = time.time() - start_time
    print(f"elapsed time is {elapsed_time} sec.")

    for error, cfg in scores[:10]:
        print(error, cfg)

    # # top score
    error_test, cfg_selected = scores[1]

    # forecast
    models = ForecastModels()
    history = data[:-30]
    yhat = models.exp_smoothing_multistep(train, 30, cfg_selected)

    yhat = yhat.tolist()
    ytrue = val.iloc[:, 0].tolist()
    error_val = measure_rmspe(yhat, ytrue)

    results = [ts_name, ts_currency, cfg_selected, error_test, error_val]

    # convert to str
    results = [str(item) for item in results]
    #save tuple list to txt file
    f = open('hw_param_retail.txt', 'a+')
    # a for append
    line = ', '.join(x for x in results)
    f.write(line + '\n')
    f.close()