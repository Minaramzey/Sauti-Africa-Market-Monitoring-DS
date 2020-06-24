import os
import pandas as pd
import psycopg2

from dotenv import load_dotenv, find_dotenv
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# from v2_dictionaries_and_lists import *
from v2_functions_and_classes import possible_product_market_pairs_for_alps, possible_product_market_pairs_for_arima_alps, product_ws_hist_ALPS_bands, product_rt_hist_ALPS_bands, product_rt_hist_arima_ALPS_bands, product_ws_hist_arima_ALPS_bands


load_dotenv()

############################################################################################################

'''Verify the credentials before running deployment. '''

############################################################################################################


def populate_bands_tables():


    #####  ALPS part  #####

    # What markets are vialables for (strong and weak) ALPS ?

    strong_candidates_retail, weak_candidates_retail, strong_candidates_wholesale, weak_candidates_wholesale = possible_product_market_pairs_for_alps()


    markets_with_problems_alps = []


    for i in range(len(strong_candidates_wholesale)):

        product_name = strong_candidates_wholesale[i][0]
        market_id = strong_candidates_wholesale[i][1]
        source_id = strong_candidates_wholesale[i][2]
        currency_code = strong_candidates_wholesale[i][3]  

        market_with_problems = product_ws_hist_ALPS_bands(product_name, market_id, source_id, currency_code, 'ALPS')

        if market_with_problems:
            markets_with_problems_alps.append(market_with_problems)

    for i in range(len(weak_candidates_wholesale)):

        product_name = weak_candidates_wholesale[i][0]
        market_id = weak_candidates_wholesale[i][1]
        source_id = weak_candidates_wholesale[i][2]
        currency_code = weak_candidates_wholesale[i][3]



        market_with_problems = product_ws_hist_ALPS_bands(product_name, market_id, source_id, currency_code, 'ALPS (weak)')

        if market_with_problems:
            markets_with_problems_alps.append(market_with_problems)



    for i in range(len(strong_candidates_retail)):

        product_name = strong_candidates_retail[i][0]
        market_id = strong_candidates_retail[i][1]
        source_id = strong_candidates_retail[i][2]
        currency_code = strong_candidates_retail[i][3]

        print(market_id)

        market_with_problems = product_rt_hist_ALPS_bands(product_name, market_id, source_id, currency_code,'ALPS')

        if market_with_problems:
            markets_with_problems_alps.append(market_with_problems)

    for i in range(len(weak_candidates_retail)):

        product_name = weak_candidates_retail[i][0]
        market_id = weak_candidates_retail[i][1]
        source_id = weak_candidates_retail[i][2]
        currency_code = weak_candidates_retail[i][3]



        market_with_problems = product_rt_hist_ALPS_bands(product_name, market_id, source_id, currency_code, 'ALPS (weak)')

        if market_with_problems:
            markets_with_problems_alps.append(market_with_problems)




    print(markets_with_problems_alps)



    #####  ARIMA based ALPS part  #####

    candidates_retail, candidates_wholesale = possible_product_market_pairs_for_arima_alps()

    markets_with_problems_arima = []

    for i in range(len(candidates_retail)):

        product_name = candidates_retail[i][0]
        market_id = candidates_retail[i][1]
        source_id = candidates_retail[i][2]
        currency_code = candidates_retail[i][3]  

        market_with_problems = product_rt_hist_arima_ALPS_bands(product_name, market_id, source_id, currency_code)

        if market_with_problems:
            markets_with_problems_arima.append(market_with_problems)


    for i in range(len(candidates_wholesale)):

        product_name = candidates_wholesale[i][0]
        market_id = candidates_wholesale[i][1]
        source_id = candidates_wholesale[i][2]
        currency_code = candidates_wholesale[i][3]  

        market_with_problems = product_ws_hist_arima_ALPS_bands(product_name, market_id, source_id, currency_code)

        if market_with_problems:
            markets_with_problems_arima.append(market_with_problems)

    print(markets_with_problems_arima)


if __name__ == "__main__":

    populate_bands_tables() 
    