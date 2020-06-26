"""test grid search on one time series"""
import context
import pandas as pd
import matplotlib.pyplot as plt
from code.data_process import DataCleaning, DataQualityCheck, TimeSeries
from code.classic_forecast import ForecastModels, ModelConfig
from code.classic_forecast import grid_search_slide_window, walk_forward_validation_slide_window
from code.classic_forecast import train_test_split, measure_rmse, score_model
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import time
# grid search one time series 
from math import sqrt, floor
import numpy as np
from joblib import Parallel, delayed
from warnings import catch_warnings, filterwarnings
from sklearn.metrics import mean_squared_error

"""get time series"""
ts = TimeSeries()

MARKET = 'Dar es salaam'
PRODUCT = 'Morogoro Rice'
SOURCE = 'EAGC-RATIN'
SALE_TYPE = 'retail'

ts.extract_data(market=MARKET, product=PRODUCT, source=SOURCE, sale_type=SALE_TYPE)

sale = ts.data
ts_name = ts.name
ts_currency = ts.currency
# interpolate for missing values
sale = sale.interpolate(method='nearest')
#sale = sale[-100:]
"""prepare data"""
# split sale into val and data
n_val = 30
val = sale[-n_val:]
data = sale[: -n_val]
# split data into train and test
n_test = round(len(data)*0.6 )
window_length = 30
slide_distance = 14
train, test = train_test_split(data, n_test)
print(f'Data length: train={len(train)}, test={len(test)}, val={len(val)}')

"""test through a series of modules for grid search """

"""model config"""
mc = ModelConfig()
# config list for one time series
cfg_list = mc.exp_smoothing_configs(seasonal=[12, 365])
print(f'total of {len(cfg_list)} configuations.')
#N=1 # example converge case
#N=80 # example non-converge case
N=97 # example of another non-converge case
"""test exp_smoothing_multistep model forecast """
# cfg = cfg_list[N]
# print(cfg)
# models = ForecastModels()
# yhat = models.exp_smoothing_multistep(train, window_length, cfg)
# print(yhat)  

"""test walk forward validation slide window method"""
# cfg = cfg_list[N]
# results = walk_forward_validation_slide_window(data, n_test, window_length, slide_distance, cfg)

"""test score model"""
# cfg = cfg_list[N]
# scores = score_model(data, n_test, window_length, slide_distance, cfg, debug=False)

"""test grid search"""
start_time = time.time() 
scores = grid_search_slide_window(data, n_test, window_length, slide_distance, cfg_list, parallel=True)
elapsed_time = time.time() - start_time
print(f"elapsed time is {elapsed_time} sec.")

for error, cfg in scores[:10]:
    print(error, cfg )

# top score
error, cfg_selected = scores[1]

results = [ts_name, ts_currency, cfg_selected, error]

# convert to str
results = [str(item) for item in results]
#save tuple list to txt file
f = open('hw_param_rice.txt', 'a+')
# a for append
line = ', '.join(x for x in results)
f.write(line + '\n')
f.close()

# prediction:
# cfg = cfg_list[N]
# print(cfg)
# models = ForecastModels()
# yhat = models.exp_smoothing_multistep(train, window_length, cfg)
# print(yhat)  