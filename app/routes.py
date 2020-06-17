import datetime
import numpy as np
import os
import pandas as pd
import psycopg2

from dotenv import load_dotenv, find_dotenv
from flask import current_app as app
from flask import json, jsonify, request

load_dotenv()

############################################################################################################

'''Verify the credentials before running deployment. '''

############################################################################################################

@app.route("/") 
def home_view(): 
        return "<h1>Welcome to Sauti DS</h1>"


@app.route('/verifyconn', methods=['GET'])
def verify_db_conn():
    '''
    Verifies the connection to the db.
    '''
    try:

        labs_conn = psycopg2.connect(user=os.environ.get('aws_db_user'),
                            password=os.environ.get('aws_db_password'),
                            host=os.environ.get('aws_db_host'),
                            port=os.environ.get('aws_db_port'),
                            database=os.environ.get('aws_db_name'))

        return 'Connection verified.'

    except:

        return 'Connection failed.'

    finally:

        if (labs_conn):
            labs_conn.close()

@app.errorhandler(404)
def page_not_found(e):
    
    return '<h1>Error 404</h1><p> Sorry, I cannot show anything arround here.</p><img src="/static/404.png">', 404


###############################################################

#############  Pulling all the data from tables.  #############

###############################################################

@app.route("/wholesale/data-quality/")
def get_table_dqws():
        labs_conn = psycopg2.connect(user=os.environ.get('aws_db_user'),
            password=os.environ.get('aws_db_password'),
            host=os.environ.get('aws_db_host'),
            port=os.environ.get('aws_db_port'),
            database=os.environ.get('aws_db_name'))
        labs_curs = labs_conn.cursor()

        Q_select_all = """SELECT * FROM qc_wholesale_observed_price;"""
        labs_curs.execute(Q_select_all)
        print("\nSELECT * Query Excecuted.")

        rows = labs_curs.fetchall()

        df = pd.DataFrame(rows, columns=[
                "id", "market", "product", "source",
                "start", "end", "timeliness", "data_length",
                "completeness", "duplicates", "mode_D", "data_points",
                "DQI", "DQI_cat"
        ])

        Q_select_all = """SELECT * FROM markets;"""
        labs_curs.execute(Q_select_all)
        print("\nSELECT * Query Excecuted.")

        rowsM = labs_curs.fetchall()
        dfM = pd.DataFrame(rowsM, columns=["id", "market_id", "market_name", "country_code"])
        
        Q_select_all = """SELECT id, source_name FROM sources;"""
        labs_curs.execute(Q_select_all)
        print("\nSELECT * Query Excecuted.")

        rowsM = labs_curs.fetchall()
        dfS = pd.DataFrame(rowsM, columns=["id", "source_name"])

        labs_curs.close()
        labs_conn.close()
        print("Cursor and Connection Closed.")


        merged = df.merge(dfM, left_on='market', right_on='market_id')
        merged["id"] = merged["id_x"]
        merged = merged.drop(["id_x", "id_y", "market_id"], axis=1)
        merged = merged.merge(dfS, left_on='source', right_on='id')
        merged["id"] = merged["id_x"]
        merged = merged.drop(["id_x", "id_y", "source"], axis=1)
        cols = ['id', 'market_name','country_code', 'product', 'source_name', 'start', 'end', 'timeliness',
        'data_length', 'completeness', 'duplicates', 'mode_D', 'data_points',
        'DQI', 'DQI_cat']
        merged = merged[cols]
        merged['start'] = merged['start'] .apply(lambda x: datetime.date.strftime(x,"%Y-%m-%d"))
        merged['end'] = merged['end'].apply(lambda x: datetime.date.strftime(x,"%Y-%m-%d"))
        merged['price_category'] = "wholesale"
        merged['DQI'] = merged['DQI'].apply(lambda x: round(x,4) if type(x) == float else None)

        result = []
        for _, row in merged.iterrows():
                        result.append(dict(row))
        return jsonify(result)

@app.route("/retail/data-quality/")
def get_table_dqrt():
        labs_conn = psycopg2.connect(user=os.environ.get('aws_db_user'),
            password=os.environ.get('aws_db_password'),
            host=os.environ.get('aws_db_host'),
            port=os.environ.get('aws_db_port'),
            database=os.environ.get('aws_db_name'))
        labs_curs = labs_conn.cursor()

        Q_select_all = """SELECT * FROM qc_retail_observed_price;"""
        labs_curs.execute(Q_select_all)
        print("\nSELECT * Query Excecuted.")

        rows = labs_curs.fetchall()

        df = pd.DataFrame(rows, columns=[
                "id", "market", "product", "source",
                "start", "end", "timeliness", "data_length",
                "completeness", "duplicates", "mode_D", "data_points",
                "DQI", "DQI_cat"
        ])

        Q_select_all = """SELECT * FROM markets;"""
        labs_curs.execute(Q_select_all)
        print("\nSELECT * Query Excecuted.")

        rowsM = labs_curs.fetchall()
        dfM = pd.DataFrame(rowsM, columns=["id", "market_id", "market_name", "country_code"])

        Q_select_all = """SELECT id, source_name FROM sources;"""
        labs_curs.execute(Q_select_all)
        print("\nSELECT * Query Excecuted.")

        rowsM = labs_curs.fetchall()
        dfS = pd.DataFrame(rowsM, columns=["id", "source_name"])

        labs_curs.close()
        labs_conn.close()
        print("Cursor and Connection Closed.")


        merged = df.merge(dfM, left_on='market', right_on='market_id')
        merged["id"] = merged["id_x"]
        merged = merged.drop(["id_x", "id_y", "market_id"], axis=1)
        merged = merged.merge(dfS, left_on='source', right_on='id')
        merged["id"] = merged["id_x"]
        merged = merged.drop(["id_x", "id_y", "source"], axis=1)
        cols = ['id', 'market_name','country_code', 'product', 'source_name', 'start', 'end', 'timeliness',
        'data_length', 'completeness', 'duplicates', 'mode_D', 'data_points',
        'DQI', 'DQI_cat']
        merged = merged[cols]
        merged['start'] = merged['start'] .apply(lambda x: datetime.date.strftime(x,"%Y-%m-%d"))
        merged['end'] = merged['end'].apply(lambda x: datetime.date.strftime(x,"%Y-%m-%d"))
        merged['price_category'] = "retail"
        merged['DQI'] = merged['DQI'].apply(lambda x: round(x,4) if type(x) == float else None)

        result = []
        for _, row in merged.iterrows():
                        result.append(dict(row))
        return jsonify(result)

@app.route("/wholesale/price-status/")
def get_table_psws():

    labs_conn = psycopg2.connect(user=os.environ.get('aws_db_user'),
                password=os.environ.get('aws_db_password'),
                host=os.environ.get('aws_db_host'),
                port=os.environ.get('aws_db_port'),
                database=os.environ.get('aws_db_name'))
    labs_curs = labs_conn.cursor()
    
    Q_select_all = """SELECT product_name, market_name, country_code,
                        source_name, currency_code, date_price,
                        observed_price, observed_class, class_method,
                        stressness
                        FROM past_wholesale_prices;"""
    labs_curs.execute(Q_select_all)
    print("\nSELECT * Query Excecuted.")

    rows = labs_curs.fetchall()

    df = pd.DataFrame(rows, columns= [
                    "product_name", "market_name", "country_code", "source_name",
                    "currency_code", "date_price", "observed_price", 
                    "observed_class", "class_method", "stressness"
            ])
    labs_curs.close()
    labs_conn.close()
    print("Cursor and Connection Closed.")

    df['date_price'] = df['date_price'].apply(lambda x: datetime.date.strftime(x,"%Y-%m-%d"))
    df['stressness'] = df['stressness'].apply(lambda x: round(x*100,2) if type(x) == float else None)
    cols = ['country_code', 'market_name', 'product_name','date_price', 'observed_price', 'currency_code', 'observed_class', 
            'class_method', 'source_name']
    df = df[cols]
    df['price_category'] = "wholesale"

    result = []
    for _, row in df.iterrows():
            result.append(dict(row))
    
    return json.dumps(result, indent=4)

@app.route("/retail/price-status/")
def get_table_psrt():

    labs_conn = psycopg2.connect(user=os.environ.get('aws_db_user'),
                password=os.environ.get('aws_db_password'),
                host=os.environ.get('aws_db_host'),
                port=os.environ.get('aws_db_port'),
                database=os.environ.get('aws_db_name'))
    labs_curs = labs_conn.cursor()
    
    Q_select_all = """SELECT product_name, market_name, country_code,
                        source_name, currency_code, date_price,
                        observed_price, observed_class, class_method,
                        stressness
                        FROM past_retail_prices;"""
    labs_curs.execute(Q_select_all)
    print("\nSELECT * Query Excecuted.")

    rows = labs_curs.fetchall()

    df = pd.DataFrame(rows, columns= [
                    "product_name", "market_name", "country_code", "source_name",
                    "currency_code", "date_price", "observed_price", 
                    "observed_class", "class_method", "stressness"
            ])
    labs_curs.close()
    labs_conn.close()
    print("Cursor and Connection Closed.")

    df['date_price'] = df['date_price'].apply(lambda x: datetime.date.strftime(x,"%Y-%m-%d"))
    df['stressness'] = df['stressness'].apply(lambda x: round(x*100,2) if type(x) == float else None)
    cols = ['country_code', 'market_name', 'product_name','date_price', 'observed_price', 'currency_code', 'observed_class', 
            'class_method', 'source_name']
    # df = df[cols]
    df['price_category'] = "retail"

    result = []
    for _, row in df.iterrows():
            result.append(dict(row))
    return json.dumps(result, indent=4)



@app.route("/wholesale/labeled/")
def get_table_psws_labeled():

    labs_conn = psycopg2.connect(user=os.environ.get('aws_db_user'),
                password=os.environ.get('aws_db_password'),
                host=os.environ.get('aws_db_host'),
                port=os.environ.get('aws_db_port'),
                database=os.environ.get('aws_db_name'))
    labs_curs = labs_conn.cursor()
    
    Q_select_all = """SELECT product_name, market_name, country_code,
                        source_name, currency_code, date_price,
                        observed_price, observed_class, class_method,
                        stressness
                        FROM past_wholesale_prices
                        WHERE observed_class IS NOT NULL;"""
    labs_curs.execute(Q_select_all)
    print("\nSELECT * Query Excecuted.")

    rows = labs_curs.fetchall()

    df = pd.DataFrame(rows, columns= [
                    "product_name", "market_name", "country_code", "source_name",
                    "currency_code", "date_price", "observed_price", 
                    "observed_class", "class_method", "stressness"
            ])
    labs_curs.close()
    labs_conn.close()
    print("Cursor and Connection Closed.")

    df['date_price'] = df['date_price'].apply(lambda x: datetime.date.strftime(x,"%Y-%m-%d"))
    df['stressness'] = df['stressness'].apply(lambda x: round(x*100,2) if type(x) == float else None)
    cols = ['country_code', 'market_name', 'product_name','date_price', 'observed_price', 'currency_code', 'observed_class', 
            'class_method', 'source_name']
    # df = df[cols]
    df['price_category'] = "wholesale"

    result = []
    for _, row in df.iterrows():
            result.append(dict(row))
    return jsonify(result)

@app.route("/retail/labeled/")
def get_table_psrt_labeled():

    labs_conn = psycopg2.connect(user=os.environ.get('aws_db_user'),
                password=os.environ.get('aws_db_password'),
                host=os.environ.get('aws_db_host'),
                port=os.environ.get('aws_db_port'),
                database=os.environ.get('aws_db_name'))
    labs_curs = labs_conn.cursor()
    
    Q_select_all = """SELECT product_name, market_name, country_code,
                        source_name, currency_code, date_price,
                        observed_price, observed_class, class_method,
                        stressness
                        FROM past_retail_prices
                        WHERE observed_class IS NOT NULL;"""
    labs_curs.execute(Q_select_all)
    print("\nSELECT * Query Excecuted.")

    rows = labs_curs.fetchall()

    df = pd.DataFrame(rows, columns= [
                    "product_name", "market_name", "country_code", "source_name",
                    "currency_code", "date_price", "observed_price", 
                    "observed_class", "class_method", "stressness"
            ])
    labs_curs.close()
    labs_conn.close()
    print("Cursor and Connection Closed.")

    print(df.dtypes)


    df['date_price'] = df['date_price'].apply(lambda x: datetime.date.strftime(x,"%Y-%m-%d"))
    df['stressness'] = df['stressness'].apply(lambda x: round(x*100,2) if type(x) == float else None)
    df['price_category'] = "retail"

    result = []
    for _, row in df.iterrows():
            result.append(dict(row))
    return jsonify(result)


@app.route("/wholesale/labeled/latest/")
def get_table_psws_labeled_latest():

    labs_conn = psycopg2.connect(user=os.environ.get('aws_db_user'),
                password=os.environ.get('aws_db_password'),
                host=os.environ.get('aws_db_host'),
                port=os.environ.get('aws_db_port'),
                database=os.environ.get('aws_db_name'))
    labs_curs = labs_conn.cursor()
    
    Q_select_all = """SELECT product_name, market_name, country_code,
                        source_name, currency_code, date_price,
                        observed_price, observed_class, class_method,
                        stressness
                        FROM past_wholesale_prices
                        WHERE observed_class IS NOT NULL;"""
    labs_curs.execute(Q_select_all)
    print("\nSELECT * Query Excecuted.")

    rows = labs_curs.fetchall()

    df = pd.DataFrame(rows, columns= [
                    "product_name", "market_name", "country_code", "source_name",
                    "currency_code", "date_price", "observed_price", 
                    "observed_class", "class_method", "stressness"
            ])
    labs_curs.close()
    labs_conn.close()
    print("Cursor and Connection Closed.")

    list_to_drop = df[df.sort_values(by=['date_price'], ascending=False).duplicated(['product_name', 'market_name', 'source_name','currency_code'], keep='first')].index

    df = df.drop(labels = list_to_drop, axis=0)

    df['date_price'] = df['date_price'].apply(lambda x: datetime.date.strftime(x,"%Y-%m-%d"))
    df['stressness'] = df['stressness'].apply(lambda x: round(x*100,2) if type(x) == float else None)
    df['price_category'] = "wholesale"

    print(df.dtypes)

    result = []
    for _, row in df.iterrows():
            result.append(dict(row))
    return jsonify(result)

@app.route("/retail/labeled/latest/")
def get_table_psrt_labeled_latest():

    labs_conn = psycopg2.connect(user=os.environ.get('aws_db_user'),
                password=os.environ.get('aws_db_password'),
                host=os.environ.get('aws_db_host'),
                port=os.environ.get('aws_db_port'),
                database=os.environ.get('aws_db_name'))
    labs_curs = labs_conn.cursor()
    
    Q_select_all = """SELECT product_name, market_name, country_code,
                        source_name, currency_code, date_price,
                        observed_price, observed_class, class_method,
                        stressness
                        FROM past_retail_prices
                        WHERE observed_class IS NOT NULL;"""
    labs_curs.execute(Q_select_all)
    print("\nSELECT * Query Excecuted.")

    rows = labs_curs.fetchall()

    df = pd.DataFrame(rows, columns= [
                    "product_name", "market_name", "country_code", "source_name",
                    "currency_code", "date_price", "observed_price", 
                    "observed_class", "class_method", "stressness"
            ])
    labs_curs.close()
    labs_conn.close()
    print("Cursor and Connection Closed.")

    list_to_drop = df[df.sort_values(by=['date_price'], ascending=False).duplicated(['product_name', 'market_name', 'source_name','currency_code'], keep='first')].index

    df = df.drop(labels = list_to_drop, axis=0)

    df['date_price'] = df['date_price'].apply(lambda x: datetime.date.strftime(x,"%Y-%m-%d"))
    df['stressness'] = df['stressness'].apply(lambda x: round(x*100,2) if type(x) == float else None)
    df['price_category'] = "retail"

    result = []
    for _, row in df.iterrows():
            result.append(dict(row))
    return jsonify(result)



########################################################################

#############  Pulling specific product market pair data.  #############

########################################################################


@app.route('/raw/')
def query_raw_data():

    query_parameters = request.args
    product_name = query_parameters.get('product_name')
    market_name = query_parameters.get('market_name')
    country_code = query_parameters.get('country_code')
    source_name = query_parameters.get('source_name')

    labs_conn = psycopg2.connect(user=os.environ.get('aws_db_user'),
                        password=os.environ.get('aws_db_password'),
                        host=os.environ.get('aws_db_host'),
                        port=os.environ.get('aws_db_port'),
                        database=os.environ.get('aws_db_name'))

    labs_curs = labs_conn.cursor()

    if source_name:

        labs_curs.execute('''
                    SELECT id
                    FROM sources
                    WHERE source_name = %s
        ''', (source_name,))

        source_id = labs_curs.fetchall()

        if not source_id:

            return 'That source name is not in the db.'

        else:

            source_id = source_id[0][0]

    query = ''' 
            SELECT *
            FROM raw_table
            WHERE
            '''
    to_filter = []


    if product_name:
        query += ' product_name=%s AND'
        to_filter.append(product_name)
    if market_name and country_code:
        market_id = market_name + ' : ' + country_code
        query += ' market_id=%s AND'
        to_filter.append(market_id)
    if source_name:
        labs_curs.execute('''
                SELECT id
                FROM sources
                WHERE source_name = %s
        ''', (source_name,))

        source_id = labs_curs.fetchall()

        if source_id:

            source_id = source_id[0][0]
            query += ' source_id = %s AND'
            to_filter.append(source_id)
    if not (product_name and market_name and country_code):
        return page_not_found(404)

    query = query[:-4] + ';'

    labs_curs.execute(query, to_filter)

    result = labs_curs.fetchall()

    if result:

        return jsonify(result)
    
    else:
        
        return page_not_found(404)

    if labs_conn:
        
        labs_conn.close()


@app.route('/retail/')
def query_retail_data():

    query_parameters = request.args
    product_name = query_parameters.get('product_name')
    market_name = query_parameters.get('market_name')
    country_code = query_parameters.get('country_code')
    source_name = query_parameters.get('source_name')

    labs_conn = psycopg2.connect(user=os.environ.get('aws_db_user'),
                        password=os.environ.get('aws_db_password'),
                        host=os.environ.get('aws_db_host'),
                        port=os.environ.get('aws_db_port'),
                        database=os.environ.get('aws_db_name'))

    labs_curs = labs_conn.cursor()

    if source_name:

        labs_curs.execute('''
                    SELECT id
                    FROM sources
                    WHERE source_name = %s
        ''', (source_name,))

        source_id = labs_curs.fetchall()

        if not source_id:

            return 'That source name is not in the db.'

        else:

            source_id = source_id[0][0]

    query_0 = ''' 
            SELECT *
            FROM past_retail_prices
            WHERE
            '''

    query_1 = '''
            SELECT *
            FROM retail_stats
            WHERE
    '''

    to_filter = []


    if product_name:
        query_0 += ' product_name=%s AND'
        query_1 += ' product_name=%s AND'
        to_filter.append(product_name)
    if market_name and country_code:
        market_id = market_name + ' : ' + country_code
        query_0 += ' market_id=%s AND'
        query_1 += ' market_id=%s AND'
        to_filter.append(market_id)
    if source_name:
        labs_curs.execute('''
                SELECT id
                FROM sources
                WHERE source_name = %s
        ''', (source_name,))

        source_id = labs_curs.fetchall()

        if source_id:

            source_id = source_id[0][0]
            query_0 += ' source_id = %s AND'
            query_1 += ' source_id = %s AND'
            to_filter.append(source_id)
    if not (product_name and market_name and country_code):
        return page_not_found(404)

    query_0 = query_0[:-4] + ';'
    query_1 = query_1[:-4] + ';'

    labs_curs.execute(query_0, to_filter)

    result = labs_curs.fetchall()



    if result:

        df = pd.DataFrame(result, columns=['id', 'product_name','market_id','market_name', 'country_code','source_id', 'source_name', 'currency_code', 'unit_scale', 'date_price', 'observed_price', 'observed_class', 'class_method', 'forecasted_price', 'forecasted_class', 'forecasting_model', 'normal_band_limit', 'stress_band_limit', 'alert_band_limit', 'stressness', 'date_run_model'])
        df['date_price'] = df['date_price'].apply(lambda x: datetime.date.strftime(x,"%Y-%m-%d"))
        df['date_run_model'] = df['date_run_model'].apply(lambda x: datetime.date.strftime(x,"%Y-%m-%d") if x is not None else x)
        df['stressness'] = df['stressness'].apply(lambda x: round(x*100,2) if x is not None else x)
        df['stressness'] = df['stressness'].replace(np.nan, -1)
        df = df.drop(labels=['id'],axis=1)

        labs_curs.execute(query_1,to_filter)

        stats = labs_curs.fetchall()

        if stats:

            stats_dict = {'start_date' : datetime.date.strftime(stats[0][5],"%Y-%m-%d"), 'end_date': datetime.date.strftime(stats[0][6],"%Y-%m-%d"), 'Mode_D': stats[0][12], 'number_of_observations': stats[0][13], 'mean': round(stats[0][14],2), 'min_price': stats[0][16], 'max_price': stats[0][20], 'days_between_start_end': stats[0][21], 'completeness': round(stats[0][22]*100 / .7123,2), 'DQI': 'not available', 'DQI_cat': 'not available'}

            labs_curs.execute('''
            SELECT *
            FROM qc_retail_observed_price
            WHERE product_name = %s
            AND market_id = %s
            ''', (product_name,market_id))

            DQI_info = labs_curs.fetchall()

            if DQI_info:

                stats_dict['DQI'] =  round(DQI_info[0][-2],2)
                stats_dict['DQI_cat'] =  DQI_info[0][-1]

        else:

            stats_dict = {'product_data':'missing'}

        return jsonify(quality = stats_dict, history = df.to_dict('records'))

    
    else:
        
        return page_not_found(404)

    if labs_conn:
        
        labs_conn.close()

@app.route('/wholesale/')
def query_wholesale_data():

    query_parameters = request.args
    product_name = query_parameters.get('product_name')
    market_name = query_parameters.get('market_name')
    country_code = query_parameters.get('country_code')
    source_name = query_parameters.get('source_name')

    labs_conn = psycopg2.connect(user=os.environ.get('aws_db_user'),
                        password=os.environ.get('aws_db_password'),
                        host=os.environ.get('aws_db_host'),
                        port=os.environ.get('aws_db_port'),
                        database=os.environ.get('aws_db_name'))

    labs_curs = labs_conn.cursor()

    if source_name:

        labs_curs.execute('''
                    SELECT id
                    FROM sources
                    WHERE source_name = %s
        ''', (source_name,))

        source_id = labs_curs.fetchall()

        if not source_id:

            return 'That source name is not in the db.'

        else:

            source_id = source_id[0][0]

    query_0 = ''' 
            SELECT *
            FROM past_wholesale_prices
            WHERE
            '''

    query_1 = '''
            SELECT *
            FROM wholesale_stats
            WHERE
    '''

    to_filter = []


    if product_name:
        query_0 += ' product_name=%s AND'
        query_1 += ' product_name=%s AND'
        to_filter.append(product_name)
    if market_name and country_code:
        market_id = market_name + ' : ' + country_code
        query_0 += ' market_id=%s AND'
        query_1 += ' market_id=%s AND'
        to_filter.append(market_id)
    if source_name:
        labs_curs.execute('''
                SELECT id
                FROM sources
                WHERE source_name = %s
        ''', (source_name,))

        source_id = labs_curs.fetchall()

        if source_id:

            source_id = source_id[0][0]
            query_0 += ' source_id = %s AND'
            query_1 += ' source_id = %s AND'
            to_filter.append(source_id)
    if not (product_name and market_name and country_code):
        return page_not_found(404)

    query_0 = query_0[:-4] + ';'
    query_1 = query_1[:-4] + ';'

    labs_curs.execute(query_0, to_filter)

    result = labs_curs.fetchall()



    if result:

        df = pd.DataFrame(result, columns=['id', 'product_name','market_id','market_name', 'country_code','source_id', 'source_name', 'currency_code', 'unit_scale', 'date_price', 'observed_price', 'observed_class', 'class_method', 'forecasted_price', 'forecasted_class', 'forecasting_model', 'normal_band_limit', 'stress_band_limit', 'alert_band_limit', 'stressness', 'date_run_model'])
        df['date_price'] = df['date_price'].apply(lambda x: datetime.date.strftime(x,"%Y-%m-%d"))
        df['date_run_model'] = df['date_run_model'].apply(lambda x: datetime.date.strftime(x,"%Y-%m-%d") if x is not None else x)
        df['stressness'] = df['stressness'].apply(lambda x: round(x*100,2) if x is not None else x)
        df['stressness'] = df['stressness'].replace(np.nan, -1)
        df = df.drop(labels=['id'],axis=1)

        labs_curs.execute(query_1,to_filter)

        stats = labs_curs.fetchall()

        if stats:

            stats_dict = {'start_date' : datetime.date.strftime(stats[0][5],"%Y-%m-%d"), 'end_date': datetime.date.strftime(stats[0][6],"%Y-%m-%d"), 'Mode_D': stats[0][12], 'number_of_observations': stats[0][13], 'mean': round(stats[0][14],2), 'min_price': stats[0][16], 'max_price': stats[0][20], 'days_between_start_end': stats[0][21], 'completeness': round(stats[0][22]*100 / .7123,2), 'DQI': 'not available', 'DQI_cat': 'not available'}

            labs_curs.execute('''
            SELECT *
            FROM qc_wholesale_observed_price
            WHERE product_name = %s
            AND market_id = %s
            ''', (product_name,market_id))

            DQI_info = labs_curs.fetchall()

            if DQI_info:

                stats_dict['DQI'] =  round(DQI_info[0][-2],2)
                stats_dict['DQI_cat'] =  DQI_info[0][-1]

        else:

            stats_dict = {'product_data':'missing'}

        return jsonify(quality = stats_dict, history = df.to_dict('records'))

    else:
        
        return page_not_found(404)

    if labs_conn:
        
        labs_conn.close()
    

@app.route("/availablepairs/")
def get_available_pairs():

    labs_conn = psycopg2.connect(user=os.environ.get('aws_db_user'),
                password=os.environ.get('aws_db_password'),
                host=os.environ.get('aws_db_host'),
                port=os.environ.get('aws_db_port'),
                database=os.environ.get('aws_db_name'))
    
    labs_curs = labs_conn.cursor()

    all_pairs = {'retail':None, 'wholesale':None}

    labs_curs.execute('''
            SELECT country_code
            FROM countries
            ''')

    countries = labs_curs.fetchall()

    if countries:

        countries = [x[0] for x in countries]

        country_market_product_pairs = {country: None for country in countries}

        for country in countries:

        
            labs_curs.execute('''
                    SELECT market_name 
                    FROM markets
                    WHERE country_code = %s
                    ''', (country,))

            markets = labs_curs.fetchall()

            if markets:

                markets = [x[0] for x in markets]

                country_market_product_pairs[country]= {market : None for market in markets}

                retail_pairs = country_market_product_pairs.copy()
                wholesale_pairs = country_market_product_pairs.copy()

                for market in markets:

                    # retail query

                    labs_curs.execute('''
                    SELECT DISTINCT(product_name) 
                    FROM past_retail_prices
                    WHERE country_code = %s
                    AND market_name = %s
                    ''', (country,market))

                    products = labs_curs.fetchall()

                    if products:

                        products = [x[0] for x in products]

                        retail_pairs[country][market] = products
                        all_pairs['retail'] = retail_pairs
                                         

                    # wholesale query

                    labs_curs.execute('''
                    SELECT DISTINCT(product_name) 
                    FROM past_wholesale_prices
                    WHERE country_code = %s
                    AND market_name = %s
                    ''', (country,market))

                    products = labs_curs.fetchall()

                    if products:

                        products = [x[0] for x in products]

                        wholesale_pairs[country][market] = products
                        all_pairs['wholesale'] = retail_pairs

                    else:

                        del wholesale_pairs[country][market]

    keys_to_drop = []

    for sale_type in ['retail', 'wholesale']:

        for key in all_pairs[sale_type].keys():

            if not all_pairs[sale_type][key]:

                keys_to_drop.append(key)
        
        for key in keys_to_drop:

            del all_pairs[sale_type][key]
        
        keys_to_drop = []


    labs_curs.close()
    labs_conn.close()


    return jsonify(all_pairs)

