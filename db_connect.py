import psycopg2
import os
from dotenv import load_dotenv

def connect_to_supabase():
    # Load environment variables from .env file
    load_dotenv()



    # Connect to the Supabase database
    try:
       conn = psycopg2.connect(sslmode='require', dbname=os.getenv('dbname'), user=os.getenv('user'), password=os.getenv('password'), host= os.getenv('host'), port= os.getenv('port'))
       
       print("Connection to the database successful!")

       return conn
    except psycopg2.Error as e:
        
        print(f"Error connecting to the database: {e}")
        return None
