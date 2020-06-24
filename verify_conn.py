import os
import psycopg2


from dotenv import load_dotenv, find_dotenv
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT 
from psycopg2 import sql

load_dotenv()

############################################################################################################

'''Verify the credentials before running deployment. '''

############################################################################################################



connection = psycopg2.connect(user=os.environ.get('aws_db_user'),
                            password=os.environ.get('aws_db_password'),
                            host=os.environ.get('aws_db_host'),
                            port=os.environ.get('aws_db_port'))#,
                            #database=os.environ.get('db_name'))

connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

# Create the cursor.

cursor = connection.cursor()

print(connection)




#Q_create_DB = """
#        CREATE DATABASE sautidb;
#              """

#cursor.execute(sql.SQL("CREATE DATABASE sautidb"))


#cursor.execute(Q_create_DB)

#connection.commit()

cursor.close()

connection.close()

