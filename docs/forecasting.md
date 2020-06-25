# Holter Winters Exponential Smoothing with Grid Search on Univariate Time Series Dataset

## Introduction: 

Based on exploratory model performance, statistical method outperform deep learning models. So Holt-Winters Exponential Smoothing provided by Statsmodel lilbary. 

When the Holter-Winters model optimization is set to true, some of the parameters are automatically tuned during fitting. These parameters are smoothing level, smoothing slope, smoothing seasonal, and damping slope.  To find the best combination for the rest of the unoptimized parameters, we used grid search method, combined with a forward sliding window validation on our time series dataset. Grid-searched parameters are: trend, dampening, seasonality, seasonal period, Box-Cox transform, removal of the bias when fitting). The RMSPE (root mean squared percentage error) is used to score the model and smallest RMSPE indicates the best combination for all model parameters. 

## Methods description: 
1. Data preprocss. Time series sequences are preprocessed (null removed, zero removed, duplication removed) and stored in our analytical database in AWS Cloud. The test sequences are extracted from analytical database giving the unique combination of product_name, market_id, and source_id. 

2. Grid-search based on slide window forecast. First we remove the outliers and interpolate the time series so it is augamented to a day-by-day timeframe. Then we split the data into initial train, initial test, and validation periods. Then a forecast window, default window length = 30 days, slides down the end of the initial train at a default pace of 14 days per slide. The 144 combinations of model parameters are fed into the model for each window forecast and model scoring.  Finally, the best score metrics returned from all valid windows corresponds to the best model parameters for the time series tested, and each time series get its own optimized model configuration. 

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
* Example sequence: retail, Dar Es Salaam,  Morogoro Rice\
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
* Time consuming. The grid search method desires large computational power. To get a better idea of computational intensity for the grid search method, we tested one time sequence: Dar Es Salaam Morogoro Rice, which has a total length of 1760 days after interpolation. On a stand-alone machine (2.6 GHz processor, 8 GB RAM), it took 108.73 min to finish grid searching of 144 configurations; while on a virtual machine with 64 GB RAM and 16 GPU cores (an AWS EC2 instance type m5ad.4xlarge), it took 17.38 min to complete the same task. For 65 sequences, it took 11.38 hours on the stand-alone machine, and 3.06 hours using the AWS EC2 virtual machine.
   
## Future work:
1. Use DQI metric Mode_D to define resolution for each time series, instead of use universal day-by-day time frame. This customized forecast resolution feature could reduce uncertainty due to interpolation and improve model accuracy. 
   
2. For time series where a large data gap exists, the interpolation method can result in a flat line, and the search method will favor dampening the trend. Since we apply a sliding window, and select best parameters based on smallest error over all sliding steps, hopefuly this data gap effect will be larged reduced.  At this point, the data gap effect is still not clear and need further investigation.
   
3. Random search and Bayesian Optimization could reduce the computation time, and are best-suited for optimization over continuous domains. These search methods worth exploring, keeping in mind of the discrete nature of the Holter-Winters model parameter domains (except for p seasonal period, which can be set as continuous integers). 
        
