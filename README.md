# Sauti-Africa-Market-Monitoring-DS

# data-science
TEAM DOCS:
 - [Product Vision Document](https://www.notion.so/Sauti-Africa-Market-Monitoring-faea97bc20054ca389f3dcad2f80bf43)


Github Links:

- [DATA-SCIENCE](https://github.com/Lambda-School-Labs/Sauti-Africa-Market-Monitoring-DS)
 
- [FRONT-END](https://github.com/Lambda-School-Labs/Sauti-Africa-Market-Monitoring-BE)

- [BACK-END](https://github.com/Lambda-School-Labs/Sauti-Africa-Market-Monitoring-BE)

 
## **Contributers**


|[Jesús Caballero](https://github.com/CodingDuckmx)                                        |[Jing Qian](https://github.com/KyleTy1er)                                        |[Taylor Curran](https://github.com/taycurran)                                        |             
| :-----------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------: | 
|                      [<img src="https://avatars2.githubusercontent.com/u/57272996?s=460&u=7bd094ffa064db7948f3f4db3aa7664e27250366&v=4" width = "200" />](https://github.com/dougscohen)                       |                      [<img src="https://avatars2.githubusercontent.com/u/41159276?s=460&u=25cdb679bf0ac27aa0da9f8852a055eb510618f0&v=4" width = "200" />](https://github.com/qianjing2020)                       |                      [<img src="https://avatars1.githubusercontent.com/u/51762885?s=460&u=38d18476e069adca0c7eebf74c4b551675af835a&v=4" width = "200" />](https://github.com/taycurran)                       
|              [<img src="https://github.com/favicon.ico" width="15"> ](https://github.com/CodingDuckmx)                    |            [<img src="https://github.com/favicon.ico" width="15"> ](https://github.com/qianjing2020)             |           [<img src="https://github.com/favicon.ico" width="15"> ](https://github.com/taycurran)            |
| [ <img src="https://static.licdn.com/sc/h/al2o9zrvru7aqj8e1x2rzsrca" width="15"> ](https://www.linkedin.com/in/jesus-caballero-medrano/)| [ <img src="https://static.licdn.com/sc/h/al2o9zrvru7aqj8e1x2rzsrca" width="15"> ](https://www.linkedin.com/) | [ <img src="https://static.licdn.com/sc/h/al2o9zrvru7aqj8e1x2rzsrca" width="15"> ](https://www.linkedin.com/in/taylorcurranagtc/) |


## **Description**

Project description goes here

## **DS Roles**

1. ### Data Engineers

    + describe the work of the data engineers here

2. ### Machine Learning Engineers

    + describe the work of the ML engineers here



## Repo Guide

-   [verify_conn.py](https://github.com/Lambda-School-Labs/Sauti-Africa-Market-Monitoring-DS/blob/master/verify_conn.py)
  verify the connection with the database create schema and creates the whole schema of the database.
    
-   [functions_and_classes.py](https://github.com/Lambda-School-Labs/Sauti-Africa-Market-Monitoring-DS/blob/master/functions_and_classes.py)
 a script with a handful set of functions used on this project. Global Methodology - It explains the main methodology.
    

### Data Flow

1.  We pull the raw data from the Stakeholder database. There are typos or misspelling words. The first script (aws_collect_data) tries to correct that with the help of dictionaries and lists script. If some products couldn't be corrected, those logs will be dropped in the error logs table.
-  **Also, we don't manipulate the numerical data, so anyone could try to drop the outliers or numerical typos the better way they considered.**
	 -   [aws_collect_data.py](https://github.com/Lambda-School-Labs/Sauti-Africa-Market-Monitoring-DS/blob/master/aws_collect_data.py)   
2.  The second script (split_bc_drop) will try to correct decimal point misplaced, drop outliers, and prices at zero making no sense. It also divides the original table into two tables: retail and wholesale prices.
	-   [split_bc_drop.py](https://github.com/Lambda-School-Labs/Sauti-Africa-Market-Monitoring-DS/blob/master/split_bc_drop.py)   
3.  A third script (qc_tables) makes some analysis of the data and drops the results into the QC retail or QC wholesale table.
	-   [qc_tables.py](https://github.com/Lambda-School-Labs/Sauti-Africa-Market-Monitoring-DS/blob/master/qc_tables.py)    

--- NOTE --- From this point, every step is duplicated, one for wholesale and one for retail. But we’ll continue to describe in terms of retail, just know there is a duplicated wholesale counterpart for everything described below. ---
    
4.  The next script (data stats) drops some important stats from the 'cleaned' data including the test for stationary. With this info, we can know in which time series we can focus on and build the ALPS bands with we can compare the prices and have a phase to show for the end-user using bands_construction.
	 -   [data_stats.py](https://github.com/Lambda-School-Labs/Sauti-Africa-Market-Monitoring-DS/blob/master/data_stats.py)   
5.  Once we have all this, we can wrap (using update_price_tables) all the info and deliver this to the web-app to be displayed, using an API deployed on Heroku.
	-   [update_prices_table.py](https://github.com/Lambda-School-Labs/Sauti-Africa-Market-Monitoring-DS/blob/master/update_prices_table.py)
    
  

## ALPS

We chose to rely on an industry-standard index called ALPS, which was developed by the World Food Program. You may find their methodology in a PDF in this repository. Due to the raw data, that is poor in terms of length, we only could work with the ALPS for only four time-series of almost 10K. So, we had to think of another possible way. The ALPS demands at least 3 years of historical data, but I felt we should consider 4 years or more of historical data, to capture better seasonalities. On the other hand, our Stakeholder suggested us to relax the requirements. Consider less length of data and see what happens. We did that. We relaxed to 2 years of data. In the case, we named the methodology as weak ALPS.

  

## ARIMA

Even relaxing the ALPS requirements we were getting around 28 time-series with their prices labeled. We need to think in another way. So we decided to change the base ALPS which is Linear Regression to an ARIMA (forecasting). This was a success for us, we get more than 100 time-series with phases.


## Future Work
I believe the following improvements could be done:

Look for the series that are not at this time stationary, and work on trying to turn them stationary. Look for time-series whose are lacking length are try to get some historical data for those, by other means. Analyze if the ARIMA based version of the ALPS is more accurate for the stakeholder, I have this feeling. Almost all the built is done the way to be automatized, but you have maybe to polish some details and built the cron job as well. Forecast trend of the prices. DS team has built the ground level of these. We have some results using Facebook Prophet and also Holter-Winters. Check out the notebooks.

### Data quality index (DQI)

A data quality index (DQI) is calculated based on six quality dimensions to rank all time series for later forecasting. Higher the DQI value, better the data quality.

  

The DQI is defined as the weighted sum of the six transformed quality dimensions:

  

$$DQI = \sum_{i=1}^6D_iW_i$$

  

where

```

D1, W1 = tdf['data_length'], 0.6

D2, W2 = tdf['completeness'], 0.3

D3, W3 = 1-tdf['mode_D'], 0.9

D4, W4 = 1-tdf['timeliness'], 0.9

D5, W5 = 1-tdf['duplicates'], 0.3

D6, W6 = tdf['data_points'], 0.9

```

tdf means the transformed dimensions are transformed (scaled to be in the range of 0 to 1). Weights assigned to the quality dimensions are empirical.

  

The following quality index are used extracted from each time series and used in the calculation of six quality dimentions.

  

Data Length: length of collected data in terms of days. It is day difference between the start and end of the time series.

  

Timeliness: gap between the end of time seires and today in terms of days. 0 means sampling is up to today, 30 means the timestamp on the most recent data was 30 days ago.

  

Completeness: The ratio of the length of valid (not Nan) data to the length of total data in a complete day-by-day time frame. 0 means all data are Nan, 1 means data is perfectly completed.

  

Duplicates: number of data sampled on a same date. 0 means no duplicates, 10 means there are 10 data entrys are duplicates.

  

Mode D: the mode (most frequent) of sampling interval in time series in terms of days. This can be considered to be the resolution of the data. 1 means most of the samples were collected on a daily basis. 7 means most of the samples were collected on a weekly basis.

  

Data points: point of valid data, calculated as Data length x Completeness - Duplicates.

  
  
  

HOT-WINTERS

  

# Holter Winters Exponential Smoothing with Grid Search on Univariate Time Series Dataset

  

## Introduction:

  

Based on exploratory model performance, statistical method outperform deep learning models. So Holt-Winters Exponential Smoothing provided by Statsmodel lilbary.

  

When the Holter-Winters model optimization is set to true, some of the parameters are automatically tuned during fitting. These parameters are smoothing level, smoothing slope, smoothing seasonal, and damping slope. To find the best combination for the rest of the unoptimized parameters, we used grid search method, combined with a forward sliding window validation on our time series dataset. Grid-searched parameters are: trend, dampening, seasonality, seasonal period, Box-Cox transform, removal of the bias when fitting). The RMSPE (root mean squared percentage error) is used to score the model and smallest RMSPE indicates the best combination for all model parameters.

  

## Methods description:

1. Data preprocss. Time series sequences are preprocessed (null removed, zero removed, duplication removed) and stored in our analytical database in AWS Cloud. The test sequences are extracted from analytical database giving the unique combination of product_name, market_id, and source_id.

  

2. Grid-search based on slide window forecast. First we remove the outliers and interpolate the time series so it is augamented to a day-by-day timeframe. Then we split the data into initial train, initial test, and validation periods. Then a forecast window, default window length = 30 days, slides down the end of the initial train at a default pace of 14 days per slide. The 144 combinations of model parameters are fed into the model for each window forecast and model scoring. Finally, the best score metrics returned from all valid windows corresponds to the best model parameters for the time series tested, and each time series get its own optimized model configuration.

  

3. Exceptions. Poor configuration of model parameters can results in the failure at model fitting. Sometimes model fitting does not converge during a certain slide, and the window is discard duringt the grid search. For time series with very poor data quality, the model fitting fails at all windows and the grid search will return nothing.

  

## Configurations:

  

* Data split (in days):\

train (start)=692, test(start)=1038, val =30\

window length = 30, sliding distance = 14

  

* Model parameters:\

A total of 144 configurations (=3x3x2x2x2x2). \

The searched parameters *t, d, s, p, b, r* are trend type, dampening type, seasonality type, seasonal period, Box-Cox transform, removal of the bias when fitting, respectively. \

t_params = ['add', 'mul', None]\

d_params = [True, False]\

s_params = ['add', 'mul', None]\

p_params = [12, 365]\

b_params = [True, False]\

r_params = [True, False]\

The additive method is preferred when the seasonal variations are roughly constant through the series, while the multiplicative method is preferred when the seasonal variations are changing proportional to the level of the series.

  

## Model output:

* Example sequence: retail, Dar Es Salaam, Morogoro Rice\

Best config: ['add', True, 'add', 12, False, False]

(explanation: additive trend, trend dampen, add seasonal period =12, no Box-Cox transform, no removal

Root mean square percentage error (rmspe):\

5.40% for last test window

1.17% for validation\

  

* Finally, all the qc_id, time series metadata, best parameter cofiguration, model forecast for the validation period and the correspondent RMSPE will be saved to database table 'hw_params_wholesale' and 'hw_params_retail' for future reference.

  

## Pros and Cons:

Pros:

* Can be customized. User has total control over window size (days for prediciton), sliding pace, train-test-val split, and grid-search domain.

* Highly tolerent, suitable for all time series, even the flat data. Will always return the best model configuration.

* Forecast results are much more accurate comparing to Facebook Prophet forecast method.

* Adaptive. User can add own model evaluation metrics, add random search instead of grid search.

Cons:

* User needs to be familiar with python and database basics.

* Time consuming. The grid search method desires large computational power. To get a better idea of computational intensity for the grid search method, we tested one time sequence: Dar Es Salaam Morogoro Rice, which has a total length of 1760 days after interpolation. On a stand-alone machine (2.6 GHz processor, 8 GB RAM), it took 108.73 min to finish grid searching of 144 configurations; while on a virtual machine with 64 GB RAM and 16 GPU cores (an AWS EC2 instance type m5ad.4xlarge), it took 17.38 min to complete the same task.

## Future work:

1. Use DQI metric Mode_D to define resolution for each time series, instead of use universal day-by-day time frame. This customized forecast resolution feature could reduce uncertainty due to interpolation and improve model accuracy.

2. For time series where a large data gap exists, the interpolation method can result in a flat line, and the search method will favor dampening the trend. Since we apply a sliding window, and select best parameters based on smallest error over all sliding steps, hopefuly this data gap effect will be larged reduced. At this point, the data gap effect is still not clear and need further investigation.

3. Random search and Bayesian Optimization could reduce the computation time, and are best-suited for optimization over continuous domains. These search methods worth exploring, keeping in mind of the discrete nature of the Holter-Winters model parameter domains (except for p seasonal period, which can be set as continuous integers).

  


