import context
import matplotlib.pyplot as plt
from code.data_process_v2 import DataCleaning, DataQualityCheck, TimeSeries
from code.db_connect import dbConnect

# test time series
ts = TimeSeries()

MARKET = 'Dar Es Salaam : TZA'
PRODUCT = 'Morogoro Rice'
SOURCE = 1
SALE_TYPE = 'retail'

ts.extract_data(MARKET, PRODUCT, SOURCE, SALE_TYPE)

print(ts.data)
print(ts.name)
print(ts.currency)
