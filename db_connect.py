import psycopg2
import os
from dotenv import dotenv_values


def connect_to_supabase():
    # Load environment variables from .env file
    config = dotenv_values('.env')




    # Connect to the Supabase database
    try:
       conn = psycopg2.connect(dbname=config['dbname'], user=config['user'], password=config['password'], host= config['host'], port= config['port'])
       
       print("Connection to the database successful!")

       cur=conn.cursor()
       return [cur,conn]
    except psycopg2.Error as e:
        
        print(f"Error connecting to the database: {e}")
        return None
