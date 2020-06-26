
# plot the forecast using best config for one time series
import context
from context import parentdir
import os
import pandas as pd
import matplotlib.pyplot as plt
from code.data_process import DataCleaning, DataQualityCheck, TimeSeries
from code.db_connect import dbConnect
from code.forecast_models import ForecastModels
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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

# unpickle dataframe
savedpickle =  parentdir+'/data/hw_config_rice.pkl'
hw_params = pd.read_pickle(savedpickle)
cfg = hw_params.loc['hw_params'][0].strip('[]').split(', ')
def convert_str_to_boolean(str_lst):
    result=[]
    for str_i in str_lst:
        if str_i == 'True':
            result.append(bool(1))
        elif str_i == 'False':
            result.append(bool(0))
        elif str_i == 'None':
            result.append(None)
        else:
            result.append(int(str_i))
    return result

cfg = convert_str_to_boolean(cfg)

# cfg = [None, False, 'add', 12, True, False]
n_test = 360

train, test = sale[:-n_test], sale[-n_test:]
fm = ForecastModels()
pred = fm.exp_smoothing_multistep(train, test, cfg)

# optimized include smoothing level, slope, seasonal, and damping slope

#pred = hw_model.predict(start=test.index[0], end=test.index[-1])


fig = plt.figure(figsize=(20,6))
ax = plt.gca()
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(pred.index, pred, label='Holt-Winters')
ax.set_title(ts_name)
ax.set_ylabel(currency)
plt.legend(loc='best')
plt.show()



savefig = parentdir+'/static/hw_rice_retail_prediction.png'
plt.savefig(savefig, dpi=600, facecolor='w', edgecolor='w')
