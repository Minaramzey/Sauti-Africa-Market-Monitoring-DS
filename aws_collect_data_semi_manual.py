import datetime
import mysql.connector
import numpy as np
import os
import psycopg2

from dotenv import load_dotenv, find_dotenv

from v2_dictionaries_and_lists import *
from provisional_list import provisional_list

load_dotenv()

############################################################################################################

'''Verify the credentials before running deployment. '''

############################################################################################################


def populate_raw_table():

    '''  
    Pulls the raw data and puts it down in our database.
    '''
    try:

        # Stablishes connection with our db



        connection = psycopg2.connect(user=os.environ.get('aws_db_user'),
                                    password=os.environ.get('aws_db_password'),
                                    host=os.environ.get('aws_db_host'),
                                    port=os.environ.get('aws_db_port'),
                                    database=os.environ.get('aws_db_name'))

        # Create the cursor.

        cursor = connection.cursor()

        # Giving it some help.

        for idrow in provisional_list:

            # Stablishes connection with Sauti's database.

            conn = mysql.connector.connect(user=os.environ.get('sauti_db_user'), password=os.environ.get('sauti_db_password'),host=os.environ.get('sauti_db_host'), database=os.environ.get('sauti_db_name'))

            cur = conn.cursor(dictionary=True)

            cur.execute('''
                    SELECT *
                    FROM platform_market_prices2
                    WHERE id = %s
            ''', (idrow,))

            rows = cur.fetchall()

            cur.close()
            conn.close()

            if rows:

                for row in rows:

                    market = row['market'].lower().title()
                    product = row['product'].lower().title()
                    market = row['market'].lower().title()
                    country = row['country']
                    source = row['source']
                    date_price = row['date'].strftime('%Y-%m-%d')
                    retail_observed_price = row['retail']
                    wholesale_observed_price = row['wholesale']
                    currency_code = row['currency']
                    unit_scale = row['unit'].lower()


                    if market in correct_markets_dict:

                        market = correct_markets_dict[market]
                    
                    if product in correct_prod_dict:

                        product = correct_prod_dict[product]
                    
                    if unit_scale in correct_scale_dict.keys():

                        unit_scale = correct_scale_dict[unit_scale]                       

                    # Pull the source id.

                    cursor.execute('''
                                SELECT id
                                FROM sources
                                WHERE source_name = %s 
                        ''', (source,))

                    source_id = cursor.fetchall()

                    if not source_id:

                        print('This product is not in the products table', product,'.')
                        
                        error_vector = (row['product'],row['market'],row['country'],row['unit'],row['source'],row['currency'],row['date'],row['retail'],row['wholesale'],row['id'],datetime.date.today(),'product')
                        
                        query_error_log = '''
                                            INSERT INTO error_logs (
                                            product_name,
                                            market_name,
                                            country_code,
                                            unit_scale,
                                            source_name,
                                            currency_code,
                                            date_price,
                                            retail_observed_price,
                                            wholesale_observed_price,
                                            mysql_db_id,
                                            error_date,
                                            possible_error
                                            )
                                            VALUES (
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s
                                            );
                        '''

                        cursor.execute(query_error_log,error_vector)

                        connection.commit()

                        print('Error with values:', error_vector, "has been added to the error's table.")
                                                
                        break

                    else:

                        source_id = source_id[0][0]   
                    
                    # Verify the product is in the products' table.

                    cursor.execute('''
                                SELECT id
                                FROM products
                                WHERE product_name = %s 
                        ''', (product,))

                    product_id = cursor.fetchall()

                    if not product_id:

                        print('This product is not in the products table', product,'.')
                        
                        error_vector = (row['product'],row['market'],row['country'],row['unit'],row['source'],row['currency'],row['date'],row['retail'],row['wholesale'],row['id'],datetime.date.today(),'product')
                        
                        query_error_log = '''
                                            INSERT INTO error_logs (
                                            product_name,
                                            market_name,
                                            country_code,
                                            unit_scale,
                                            source_name,
                                            currency_code,
                                            date_price,
                                            retail_observed_price,
                                            wholesale_observed_price,
                                            mysql_db_id,
                                            error_date,
                                            possible_error
                                            )
                                            VALUES (
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s
                                            );
                        '''

                        cursor.execute(query_error_log,error_vector)

                        connection.commit()

                        print('Error with values:', error_vector, "has been added to the error's table.")
                                                
                        break


                    # Verify the market is in the markets' table.

                    cursor.execute('''
                                SELECT market_id
                                FROM markets
                                WHERE market_name = %s 
                                AND country_code = %s
                        ''', (market,country))

                    market_id = cursor.fetchall()

                    if not market_id:

                        print('This market', market, 'is not in the country', country,'.')
                        
                        error_vector = (row['product'],row['market'],row['country'],row['unit'],row['source'],row['currency'],row['date'],row['retail'],row['wholesale'],row['id'],datetime.date.today(),'market')
                        
                        query_error_log = '''
                                            INSERT INTO error_logs (
                                            product_name,
                                            market_name,
                                            country_code,
                                            unit_scale,
                                            source_name,
                                            currency_code,
                                            date_price,
                                            retail_observed_price,
                                            wholesale_observed_price,
                                            mysql_db_id,
                                            error_date,
                                            possible_error
                                            )
                                            VALUES (
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s
                                            );
                        '''

                        cursor.execute(query_error_log,error_vector)

                        connection.commit()

                        print('Error with values:', error_vector, "has been added to the error's table.")
                        
                        break

                    else:
                        market_id = market_id[0][0]

                    # Verify the currency is in the currencies' table.

                    cursor.execute('''
                                SELECT id
                                FROM currencies
                                WHERE currency_code = %s 
                        ''', (currency_code,))

                    currency_id = cursor.fetchall()[0][0]

                    if not currency_id:

                        print('This currency is not in the db', currency_code,'.')

                        error_vector = (row['product'],row['market'],row['country'],row['unit'],row['source'],row['currency'],row['date'],row['retail'],row['wholesale'],row['id'],datetime.date.today(),'currency')
                        
                        query_error_log = '''
                                            INSERT INTO error_logs (
                                            product_name,
                                            market_name,
                                            country_code,
                                            unit_scale,
                                            source_name,
                                            currency_code,
                                            date_price,
                                            retail_observed_price,
                                            wholesale_observed_price,
                                            mysql_db_id,
                                            error_date,
                                            possible_error
                                            )
                                            VALUES (
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s
                                            );
                        '''

                        cursor.execute(query_error_log,error_vector)

                        connection.commit()

                        print('Error with values:', error_vector, "has been added to the error's table.")

                        break

                    vector = (product,market_id,unit_scale,source_id,currency_code,date_price, retail_observed_price,wholesale_observed_price)

                    # Verify if the vector is already in the db:

                    cursor.execute('''
                            SELECT id
                            FROM raw_table
                            WHERE product_name = %s 
                            AND market_id = %s 
                            AND unit_scale = %s 
                            AND source_id = %s 
                            AND currency_code = %s 
                            AND date_price = %s 
                    ''', (product,market_id,unit_scale,source_id,currency_code,date_price))                

                    result = cursor.fetchall()

                    if not result: 

                        query_insert_product_info = '''
                                            INSERT INTO raw_table (
                                            product_name,
                                            market_id,
                                            unit_scale,
                                            source_id,
                                            currency_code,
                                            date_price,
                                            retail_observed_price,
                                            wholesale_observed_price
                                            )
                                            VALUES (
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s,
                                                %s
                                            );
                        '''

                        cursor.execute(query_insert_product_info,vector)

                        connection.commit()

                        print(vector, 'added to the db.')

                    else:
                        
                        # print(vector, 'already in the db.')
                        pass

                    del result

        # cursor.close()
        # connection.close()

    except (Exception, psycopg2.Error) as error:
        print('Error inserting or verifying the values.')


        error_vector = (row['product'],row['market'],row['country'],row['unit'],row['source'],row['currency'],row['date'],row['retail'],row['wholesale'],row['id'],datetime.date.today(),'unknown')
        
        query_error_log = '''
                            INSERT INTO error_logs (
                            product_name,
                            market_name,
                            country_code,
                            unit_scale,
                            source_name,
                            currency_code,
                            date_price,
                            retail_observed_price,
                            wholesale_observed_price,
                            mysql_db_id,
                            error_date,
                            possible_error
                            )
                            VALUES (
                                %s,
                                %s,
                                %s,
                                %s,
                                %s,
                                %s,
                                %s,
                                %s,
                                %s,
                                %s,
                                %s,
                                %s
                            );
        '''

        cursor.execute(query_error_log,error_vector)

        connection.commit()

        print('Error with values:', error_vector, "has been added to the error's table.")

    finally:

        if (connection):
            cursor.close()
            connection.close()
            print('Connection closed.')

def write_index():

    # Stablishes connection with the Stakeholder's db.

    conn = mysql.connector.connect(user=os.environ.get('sauti_db_user'), password=os.environ.get('sauti_db_password'),host=os.environ.get('sauti_db_host'), database=os.environ.get('sauti_db_name'))

    cur = conn.cursor()

    cur.execute('''
            SELECT MAX(id)
            FROM platform_market_prices2
    ''')

    final_id = cur.fetchall()

    if final_id:

        final_id = final_id[0][0]


        connection = psycopg2.connect(user=os.environ.get('aws_db_user'),
                                    password=os.environ.get('aws_db_password'),
                                    host=os.environ.get('aws_db_host'),
                                    port=os.environ.get('aws_db_port'),
                                    database=os.environ.get('aws_db_name'))

        # Create the cursor.

        cursor = connection.cursor()

        # Define the values to be inserted.

        pulled_data = datetime.datetime.now()

        vector(pulled_data,final_id)

        query_logs = '''
                            INSERT INTO pulling_logs (
                            pulled_date,
                            final_id
                            )
                            VALUES (
                                %s,
                                %s
                            );
        '''

        cursor.execute(query_logs,vector)

        connection.commit()




if __name__ == "__main__":

    populate_raw_table()
    # write_index()