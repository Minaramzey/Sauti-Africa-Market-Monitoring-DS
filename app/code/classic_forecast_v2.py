"""grid search one time series 
v2 works for our aws analytical db
also added square root mean percentage error metrics"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from math import sqrt, ceil
from joblib import Parallel, delayed
from warnings import catch_warnings, filterwarnings
from sklearn.metrics import mean_squared_error

class ForecastModels:
    def __init__(self):
        pass

    def simple_forecast(self, history, config):
        # one-step baseline models
        n, offset, avg_type = config
        """
        n: number of observation in history used for forecast
        offset: seasonality
        avg_type: how to average the predictions"""
        if avg_type == 'persist':
            # naive method
            return history[n]  # observation n
        # collect values to average
        values = list()
        if offset == 1:
            # no seasonality
            values = history[n:]  # last n observations
        else:
            if n * offset > len(history):
                # skip bad configs
                raise Exception(
                    f'Config beyond end of data: {n: %d} *{offset: %d} > {len(history)}')
            for i in range(1, n + 1):
                # try and collect n values using offset
                idx = i * offset
                values.append(history[idx])  # last n observations spaced by offset
        # check if we can average
        if len(values) < 2:
            raise Exception('Cannot calculate average')
        # mean of last n values
        if avg_type == 'mean':
            return np.mean(values)
        # median of last n values
        return np.median(values)

    def exp_smoothing_onestep(self, history, config):
        # one-step Holt Winter's Exponential Smoothing forecast
        t, d, s, p, b, r = config
        history = np.array(history)
        model = ExponentialSmoothing(
            history, trend=t, damped=d, seasonal=s, seasonal_periods=p)

        model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
        # make one step forecast
        yhat = model_fit.predict(len(history), len(history))
        return yhat[0]

    def exp_smoothing_multistep(self, train, n_output, cfg):
        """
        input: train: historical time series
            window_L: prediction length
            config: model configuration
        output: multipl-step Holt Winter's Exponential Smoothing forecast
        """
        
        t, d, s, p, b, r = cfg
        start = len(train)
        end = len(train)+n_output-1
        #in cfg: trend type, dampening type, seasonality type, seasonal period, Box-Cox transform, removal of the bias when fitting    
             
        try:
            model = ExponentialSmoothing(train, trend=t, damped=d, seasonal=s, seasonal_periods=p)
            model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
            #print(cfg)
            #print('Converge succceed!')
            yhat = model_fit.predict(start=start, end=end)
            yhat = round(yhat, 2)
            
        except:
            #print('Converge Problem. Poor config skipped.')
            yhat = np.empty(n_output, )
            yhat[:] = np.nan

        return yhat

class ModelConfig:
    # model configurations, options for hyperparameters
    def __init__(self):
        pass

    def simple_configs(self, max_length, seasonal=[1]):
        # create a set of configs, offset is seasonality
        configs = list()
        for i in range(1, max_length + 1):
            # number of observation used as history
            for o in seasonal:
                for t in ['persist', 'mean', 'median']:
                    cfg = [i, o, t]
                    configs.append(cfg)
        return configs
        
    def exp_smoothing_configs(self, seasonal=[None]):
        # in cfg: trend type, dampening type, seasonality type, seasonal period, Box-Cox transform, removal of the bias when fitting
        configs = list()
        t_params = ['add', 'mul', None]
        d_params = [True, False]
        s_params = ['add', 'mul', None]
        p_params = seasonal
        b_params = [True, False]
        r_params = [True, False]
        # create config instances 
        for t in t_params:
            for d in d_params:
                for s in s_params:
                    for p in p_params:
                        for b in b_params:
                            for r in r_params:
                                cfg = [t, d, s, p, b, r]
                                configs.append(cfg)
        return configs

  
def measure_rmse(y_true, y_pred):
    # root mean squared error or rmse
    rmse_est = round(sqrt(mean_squared_error(y_true, y_pred)), 2)    
    return rmse_est

def measure_rmspe(y_true, y_pred):
    # root mean sqared percentage error
    r1 = np.divide(y_pred, y_true) # cautions! only apply when y is not zero
    r2 = np.ones(len(y_true))
    rmspe_est = round(100*sqrt(mean_squared_error(r1, r2)), 2)    
    return rmspe_est

def train_test_split(data, n_test):
    """split a univariate dataset into train/test sets
    can be pd series or numpy array"""
    return data[:-n_test], data[-n_test:]

def walk_forward_validation_one_step(data, n_test, cfg):
    """
    walk forward one step validation
    """    
    models = ForecastModels()
    predictions = list()

    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    for i in range(len(test)):
        # step over each time-step in test dataset
        """ change the model here:"""
        yhat = models.exp_smoothing_onestep(history, cfg)
        predictions.append(yhat)
        # dump the finished test to history
        history.append(test[i])
    
    error = measure_rmspe(test, predictions)
    return error, predictions

def walk_forward_validation_slide_window(data, n_test, window_length,  slide_distance, cfg):
    """walk forward validation based on sliding window
    window_length: a window of multistep preditions
    slidingLength: steps to forward during each slide
    """
    models =ForecastModels()
    # fine-ajust n_test so it will round up on selected sliding window
    n_windows = ceil((n_test -  window_length) / slide_distance)  
    n_test = n_windows * slide_distance + window_length
    # print(f'Ajusted n_test = {n_test}')
    # split dataset
    train, test = train_test_split(data, n_test)
    
    # initiate slide number
    i = 0
    # use list to initiate prediction  
    pred = [] 
       
    while i < n_windows:
        # slide over first n-1 windows                     
        yhat = models.exp_smoothing_multistep(train, slide_distance, cfg).tolist()
        # update the train with test (n = slide_distance) once prediction is done 
        pred.append(yhat)
        train.append(test[(i * slide_distance) : ((i + 1) * slide_distance - 1)])   
        i = i + 1
    
    # last window predict the entire window length
    yhat = models.exp_smoothing_multistep(train, window_length, cfg).tolist()
    pred.append(yhat) # for each time series, pred is an array of nan or valid forecasts
    
    # return VALID forecast and error ONLY
    # pred is nested list, we need to flatten it 
    predictions = []
    # now flatten the nested list (apply to both valid forecast and nan)
    for sublist in pred:
        for item in sublist:
            predictions.append(item)       
    
    if sum(np.isnan(predictions)) == 0:
        # only return valid forecast
        error = measure_rmspe(test.iloc[:,0].tolist(), predictions)
    else:
        error = np.nan     
    return (error, predictions)
    
      
def score_model(data, n_test, window_length, slide_distance, cfg, debug=True):
    # score a model, return None on failure, else return rmspe
    error = []
    # init result
    result = ()
    if debug:
        # show all warnings and fail on exception when debugging
        error, _ = walk_forward_validation_slide_window(data,  n_test, window_length, slide_distance, cfg)
    else:		
        try:
            # one failure during model validation suggests an unstable config (causing converge failure)
            with catch_warnings():
                # never show warnings when grid searching, too noisy
                filterwarnings("ignore")
                error, _ = walk_forward_validation_slide_window(data, n_test, window_length, slide_distance, cfg)
        except:
            error = None
    if error is not None:
        # check for an interesting result
        result = (error, cfg)
        # pair up config and error
    return result

def grid_search_one_step(data, n_test, cfg_list, parallel=True):
    # grid search using one step forward walk validation method
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=-1, backend='loky')
        tasks = (delayed(score_model)(data, n_test, window_length, slide_distance, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, n_test, window_length, slide_distance, cfg) for cfg in cfg_list]
    # remove empty results: None if there is an error for scoring rmspe
    scores = [r for r in scores if r[1] != None]
    # sort by error, ascending
    scores.sort(key=lambda tup: tup[1])
    return scores

def grid_search_slide_window(data, n_test, window_length, slide_distance, cfg_list, parallel=True):
    # grid search using the window_slide method
    scores = []
    if parallel:
        #breakpoint()
        # execute configs in parallel
        executor = Parallel(n_jobs=-1, backend='loky')
        tasks = (delayed(score_model)(data, n_test, window_length, slide_distance, cfg) for cfg in cfg_list)
        scores = executor(tasks)
        
    else:
        #scores = [score_model(data, n_test, window_length, slide_distance, cfg) for cfg in cfg_list]
        for i, cfg in enumerate(cfg_list):
            print((i, cfg))
            score = score_model(data, n_test, window_length, slide_distance, cfg)
            scores.append(score)
    
    # clean up score_list: remove nan error entry
    score_lst=[]
    for r in scores:
        if np.isnan(r[0]) == False:
            score_lst.append(r)
    #score_lst = [r for r in scores if np.isnan(r[1])==False]
    # sort by error, ascending (aka, best cfg on top)
    score_lst.sort(key=lambda tup: tup[0])
    return score_lst

if __name__ == '__main__':
    # just do a simple test run with dummy data
    data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    n_test = 10 # number of observation used for test    
    # model configs
    min_length = len(data) - n_test  # max length used as history    
    mc = ModelConfig()
    cfg_list = mc.simple_configs(min_length=max_length, seasonal=[1])
    # grid search
    scores = grid_search_one_step(data, cfg_list, n_test)
    # list top 10 configs
    for cfg, error in scores[:10]:
        print(cfg, error)
        


