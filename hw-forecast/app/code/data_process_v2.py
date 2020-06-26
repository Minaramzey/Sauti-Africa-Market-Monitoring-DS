# In data_process_v2 is tailored for analytical database table. So raw_table has clean data, no duplication, but still need to address outliers
import os
import sys
currentdir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, currentdir)
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import copy
from datetime import datetime, date
pd.set_option('display.max_columns', 20)
from db_connect import dbConnect

class DataCleaning:
    """ method to clean data, apply to the whole data set (mixed time series)"""
    def __init__(self):
        pass
        
    def read_data(self, data=None):
        if data is None:
            print("Warning: No data provided")
        if (isinstance(data, pd.DataFrame) == False):
            print( "Input should be a dataframe!")
        else:
            df = pd.DataFrame(data)
            print('Data is fed to class object.')
        return df
              
    def remove_zeros(self, data):
        """clean all invalid entries
        cost cannot be 0, replace zeros with NaN"""
        df = data.copy()
        cols = ['wholesale', 'retail']
        
        df[cols] = df[cols].replace({0: np.nan})
        if np.prod(df['wholesale'] != 0):
            print('All zero values has been replaced with NaN successfully')
        else:
            print('Zero to NaN process not complete.')
        return df    
          
        # # remove str in wholesale retail columns
        # str_to_remove_list = ['Wholesale', 'retail','NaN']
        # df[df['wholesale'].isin(str_to_remove_list)] = np.NaN
        # df[df['retail'].isin(str_to_remove_list)] = np.NaN       

    def convert_dtypes(self, data):
        """change each column to desired data type"""
        df = data.copy()
        # # change date to datetime
        # df['date'] = pd.to_datetime(df['date'])

        # change num dtype to float
        df['wholesale'] = df['wholesale'].astype('float')
        df['retail'] = df['retail'].astype('float')
      
        # change text col to categorical
        str_cols = ['market', 'product', 'country', 'currency']
        for item in str_cols:
            df[item] = df[item].astype('category')
        
        print('Data type converted. Numericals converted to float, date to datatime type, and non-numericals to category.')
       
        return df
     

class DataQualityCheck:
    """contain methods for quality check for one time series"""
    def __init__(self):
        pass
    
    def read_data(self, data=None):
        if data is None:
            print("Warning: No data provided")
        if (isinstance(data, pd.Series) == False) & (isinstance(data.index, pd.DatetimeIndex)):
            print("Data needs to be pandas series with datetime index!")
        else:
            df = pd.Series(data)
        return df

    def remove_outliers(self, df, lower=0.05, upper=0.85):
        """remove outliers from a series"""
        y = df.copy()
        lower_bound, upper_bound = y.quantile(lower), y.quantile(upper)
        
        idx = y.between(lower_bound, upper_bound)
        y = y[idx]
        return y
        
    def remove_duplicates(self, df):
        """remove duplicated rows, keep the first, run after remove_outlier!!!"""
        y = df.copy()
        rows_rm = y.index.duplicated(keep='first')
        if np.sum(rows_rm):
            y = y[~rows_rm]
        return y  

    def day_by_day(self, df):
        """construct time frame and create augumented time series"""
        y = df.copy()
        
        START, END = y.index.min(), y.index.max()        
        # construct a time frame from start to end
        date_range = pd.date_range(start=START, end=END, freq='D')
        time_df = pd.DataFrame([], index=date_range)
        # this is time series framed in the complete day-by-day timeframe
        y_t = time_df.merge(y, how='left', left_index=True, right_index=True)
        return y_t


class TimeSeries():
    """genereate one time series data (cleaned) for analysis"""
    def __init__(self):
        pass
    
    def extract_data(self, market_id, product_name, source_id, sale_type):
        """extract time series from raw table in aws database
        input str of market information
        output clean time series data with unique currency 
        """        
        ts_name = sale_type + ', product: ' + product_name + ', market: ' + market_id + ', source: ' + str(source_id)
            
        db_c = dbConnect()
        raw_data = db_c.read_analytical_db('raw_table')
        # ## Clean data
        # dc = DataCleaning()        
        # df  = dc.read_data(raw_data)
        # df = dc.remove_zeros(df)
        # df = dc.convert_dtypes(df)
        df = raw_data.copy()
        # create subset for testing DataQualityCheck module
        cond1 = (df['product_name']==product_name)
        cond2 = (df['source_id']==source_id)
        cond3 = (df['market_id']==market_id)

        subset = df[cond1 & cond2 & cond3].sort_values(by='date_price', ascending=True).set_index('date_price')
        currency = df['currency_code'].unique()

        # this is the sale time series
        sale_colname = sale_type+'_observed_price'
        sale = subset[[sale_colname, 'currency_code']]

        if len(currency) > 1:
            # inspect in same time series but different currency, select the one with most data
            len_lst = []
            for C in currency:
                sale_x = sale.loc[sale['currency_code'] == C, sale_colname]
                len_lst.append(len(sale_x))
        
            currency = currency[len_lst.index(max(len_lst))]
            sale = sale.loc[sale['currency_code'] == currency,:]
        
        #get only pd series for sale data for deeper clearning
        sale = sale[sale_colname]
        
        last_date = sale.index[-1]
        last_observation = sale.values[-1]
        # time series clean up
        dqc = DataQualityCheck()
        sale = dqc.remove_outliers(sale, 0.05, 0.8)
        sale = dqc.remove_duplicates(sale)
        sale = dqc.day_by_day(sale)
        
        self.name = ts_name
        self.currency = currency
        self.data = sale
        self.lastDate = last_date
        self.lastObservation = last_observation
        # plt.plot(sale,'.')
        # plt.show()
        # return sale

  
