"""
test grid search on one time series
v2 use aws analytical db instead of stakeholder db"""
import context
import pandas as pd
# import matplotlib.pyplot as plt
# from statsmodels.tsa.holtwinters import ExponentialSmoothing
import time
from math import sqrt, floor
import numpy as np
# from joblib import Parallel, delayed
# from warnings import catch_warnings, filterwarnings
from code.data_process_v2 import DataCleaning, DataQualityCheck, TimeSeries
from code.classic_forecast_v2 import ForecastModels, ModelConfig, grid_search_slide_window, walk_forward_validation_slide_window, train_test_split, score_model, measure_rmspe

"""get time series"""
ts = TimeSeries()

MARKET = 'Dar Es Salaam : TZA'
PRODUCT = 'Morogoro Rice'
SOURCE = 1
SALE_TYPE = 'retail'

ts.extract_data(MARKET, PRODUCT, SOURCE, SALE_TYPE)
sale = ts.data
ts_name = ts.name
ts_currency = ts.currency
# interpolate for missing values
sale = sale.interpolate(method='nearest')
#sale = sale[-100:] # shorten for quick test only

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
N=1 # example converge case
#N=80 # example non-converge case
#N=97 # example of another non-converge case

"""test exp_smoothing_multistep model forecast """
# cfg = cfg_list[N]
# print(cfg)
# models = ForecastModels()
# yhat = models.exp_smoothing_multistep(train, window_length, cfg)
# print(yhat)  

"""test walk forward validation slide window method"""
# cfg = cfg_list[N]
# results = walk_forward_validation_slide_window(data, n_test, window_length, slide_distance, cfg)
# print(results)
# breakpoint()

"""test score model"""
# cfg = cfg_list[N]
# scores = score_model(data, n_test, window_length, slide_distance, cfg, debug=False)

# """test grid search"""
start_time = time.time() 
scores = grid_search_slide_window(data, n_test, window_length, slide_distance, cfg_list, parallel=True)
elapsed_time = time.time() - start_time
print(f"elapsed time is {elapsed_time} sec.")

for error, cfg in scores[:10]:
    print(error, cfg )

# # top score
error_test, cfg_selected = scores[1]

#error_test, cfg_selected = [5.4, ['add', True, 'add', 12, False, True]]

# forecast
models = ForecastModels()
history = data[:-30]
yhat = models.exp_smoothing_multistep(train, 30, cfg_selected)


yhat = yhat.tolist()
ytrue = val.iloc[:,0].tolist()
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

