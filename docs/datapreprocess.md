### Data cleaning

### Data quality index (DQI)

A data quality index (DQI) is calculated based on six quality dimensions to rank all time series for later forecasting. Higher the DQI value, better the data quality.

The DQI is defined as the weighted sum of the six transformed quality dimensions:

$$DQI = \sum_{i=1}^6D_iW_i$$

where  

    D1, W1 = tdf['data_length'], 0.6
    D2, W2 = tdf['completeness'], 0.3
    D3, W3 = 1-tdf['mode_D'], 0.9
    D4, W4 = 1-tdf['timeliness'], 0.9
    D5, W5 = 1-tdf['duplicates'], 0.3
    D6, W6 = tdf['data_points'], 0.9

tdf means the transformed dimensions are transformed (scaled to be in the range of 0 to 1). Weights assigned to the quality dimensions are empirical. 

The following quality index are used extracted from each time series and used in the calculation of six quality dimentions. 

Data Length: length of collected data in terms of days. It is day difference between the start and end of the time series. 

Timeliness: gap between the end of time seires and today in terms of days. 0 means sampling is up to today, 30 means the timestamp on the most recent data was 30 days ago.

Completeness: The ratio of the length of valid (not Nan) data to the length of total data in a complete day-by-day time frame. 0 means all data are Nan, 1 means data is perfectly completed.

Duplicates: number of data sampled on a same date. 0 means no duplicates, 10 means there are 10 data entrys are duplicates.   

Mode D: the mode (most frequent) of sampling interval in time series in terms of days. This can be considered to be the resolution of the data. 1 means most of the samples were collected on a daily basis. 7 means most of the samples were collected on a weekly basis.

Data points: point of valid data, calculated as Data length x Completeness - Duplicates.
   