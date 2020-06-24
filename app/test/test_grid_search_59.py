"""grid-search for all database series using holter-winter method"""
import context
import time
# from math import sqrt, floor
# import numpy as np
# import pandas as pd
from code.db_connect import dbConnect
from code.data_process_v2 import DataCleaning, DataQualityCheck, TimeSeries
from code.classic_forecast_v2 import ForecastModels, ModelConfig, grid_search_slide_window, walk_forward_validation_slide_window, train_test_split, score_model, measure_rmspe
from test_59_sequence import ts_list

## test time series
SALE_TYPE = 'retail'

## get source id map
db_c = dbConnect()
source_tablename = 'sources'
source_table = db_c.read_analytical_db(source_tablename)

"""model config"""
mc = ModelConfig()
# config list for one time series
cfg_list = mc.exp_smoothing_configs(seasonal=[6, 12])
print(f'A total of {len(cfg_list)} configurations.')

start_time = time.time()

ts = TimeSeries()
for tsi in ts_list:

    # get right parameters for ts object
    MARKET = tsi[1]
    PRODUCT = tsi[0]
    SOURCE = tsi[3]
    print(MARKET, PRODUCT, SOURCE)
    try:
        ts.extract_data(MARKET, PRODUCT, SOURCE, SALE_TYPE)
        sale = ts.data
        ts_name = ts.name
        ts_currency = ts.currency
        last_date = ts.lastDate
        last_observation = ts.lastObservation
    except:
        print('Cannot extract time sequence')
    try:
        # interpolate for missing values
        sale = sale.interpolate(method='nearest')
    except:
        print('Cannot interpolate using nearest method.')
    try: 
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
    except:
        print('Cannot split data to prescribed split length!')
    try:    
        scores = grid_search_slide_window(
            data, n_test, window_length, slide_distance, cfg_list, parallel=True)
        elapsed_time = time.time() - start_time
        

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
        avg_yhat = np.mean(yhat)

        results = [ts_name, ts_currency, last_date, last_observation, cfg_selected, error_val, avg_yhat, yhat]

        # convert to str
        results = [str(item) for item in results]
        #save tuple list to txt file
        f = open('hw_param_retail_59.txt', 'a+')
        # a for append
        line = ', '.join(x for x in results)
        f.write(line + '\n')
        f.close()
    except:
        print(f"{MARKET}, {PRODUCT}, {SOURCE} behaved abnormally during grid search. Further investigation needed. ")    
print(f"Total elapsed time is {elapsed_time} sec for {len}.")
