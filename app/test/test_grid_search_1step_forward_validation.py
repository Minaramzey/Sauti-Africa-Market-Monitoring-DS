"""test grid search on one time series"""
import context
import pandas as pd
import matplotlib.pyplot as plt
from code.grid_search import grid_search
from code.forecast_models import ModelConfig
from code.data_process import DataCleaning, DataQualityCheck, TimeSeries

# get time series
ts = TimeSeries()

MARKET = 'Dar es salaam'
PRODUCT = 'Morogoro Rice'
SOURCE = 'EAGC-RATIN'
SALE_TYPE = 'retail'

ts.extract_data(market=MARKET, product=PRODUCT, source=SOURCE, sale_type=SALE_TYPE)

sale = ts.data

# interpolate for missing values
sale = sale.interpolate(method='nearest')

ts_name = ts.name
currency = ts.currency

# grid search
data = sale.to_numpy().flatten()

# number of observation used for test in forward validation
n_test = 360*4 
# minimum length used as history
min_length = len(data) - n_test  

mc = ModelConfig()
# config list for one time series
cfg_list = mc.exp_smoothing_configs(seasonal=[0, 12])
print(f'Grid search on a total of {len(cfg_list)} configurations')
print("------------"*5)
print('Start grid searching, now off your chair and go jumping jack...')
# grid search
scores = grid_search(data, cfg_list, n_test)

# list top 1 configs
# for cfg, error in scores[:10]:
#     print(cfg, error)
# top score
cfg_selected, error = scores[1]
result = [cfg_selected, error]
breakpoint()

save_data = pd.DataFrame([result[0], result[1]], columns=['hw_params', 'RMSE'])
#'t', 'd', 's', 'p', 'b', 'r'])
## columns: qc_id, trend type, dampening type, seasonality type, seasonal period, Box-Cox transform, removal of the bias when fitting

# save to csv file
savecsv = 'hw_config_rice' +'.csv'
save_data.to_csv(savecsv)

# pickle dataframe
savepickle =  'hw_config_rice'+'.pkl'
save_data.to_pickle(savepickle)
