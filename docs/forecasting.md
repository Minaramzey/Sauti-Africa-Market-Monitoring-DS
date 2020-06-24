Based on exploratory model performance, statistical method outperform deep learning models. 

So Holt-Winters Exponential Smoothing provided by Statsmodel lilbary is used with grid search methodology to find the model's best parameters on our time series data.

When the model optimization is set to true for the Stats Holt-Winter's method, some of the parameters are automatically tuned during fitting. These parameters are smoothing level, smoothing slope, smoothing seasonal, and damping slope. 

The following process describes how to gird search for other parameters (trend, dampening, seasonality, seasonal period, Box-Cox transform, removal of the bias when fitting). The RMSPE (root mean squared percentage error) is used to evaluate model performance. 

The computation is intensive.  For time series Dar Es Salaam Morogoro Rice (a total length of 1760 day after interpolation), it took 108.73 min to finish grid searching of 144 configurations  on a 2.6 GHz processor, 8 GB RAM computer while Spotify music is on. While using virtual machine with 64 GB RAM and 16 GPU cores (an AWS EC2 instance type m5ad.4xlarge), it took 17.38 min to complete the same task. 

The following specifies the configuration during grid searching and model output. 

Data length in days:\
train (start)=692, test(start)=1038, val =30
window length = 30, sliding distance = 14

Model parameters:\
A total of 144 configurations (=3x3x2x2x2x2). The searched parameters t, d, s, p, b, r are trend type, dampening type, seasonality type, seasonal period, Box-Cox transform, removal of the bias when fitting, respectively. Sometimes the methods could not converge on a certain poor model configuration. 

t_params = ['add', 'mul', None]\
d_params = [True, False]\
s_params = ['add', 'mul', None]\
p_params = [12, 365]\
b_params = [True, False]\
r_params = [True, False]

The additive method is preferred when the seasonal variations are roughly constant through the series, while the multiplicative method is preferred when the seasonal variations are changing proportional to the level of the series. 

Model output:\
retial, Dar Es Salaam,  Morogoro Rice\
Best config: ['add', True, 'add', 12, False, False]
(explanation: additive trend, trend dampen, add seasonal period =12, no Box-Cox transform, no removal
Root mean square percentage error (rmspe)\
5.40% for last test window
1.15% for validation

Finally, all the qc_id, best parameters, and RMSPE are save to database table 'hw_params_wholesale' and 'hw_params_retail' for future reference. 

Pros and Cons:

Pros: 
1. Can be customized. User has total control over window size (days for prediciton), train-test-val split, and grid to start/refine search. 
3. Highly tolerent, suitable for all time series, even the flat data. Will always return the best model configuration. 
4. Forecast results are highly accurate. 
5. Adaptive. User can add own model evaluation metrics, add random search instead of grid search.
   
Cons: 
1. Time consuming
2. User need to be familiar with python basics.
   
Desired Model Improvements:
1. Use DQI metric Mode_D to define resolution for each time series, instead of use universal day-by-day time frame. 
2. For time series where a large data gap exists, the interpolation method can result in a flat line, and the model will try to dampen the trend. Since we apply a sliding window, and select best parameters based smallest error over all sliding steps, hopefuly this data gap effect will be larged reduced.  At this point, the data gap effect is not clear and need further investigation. 
        
