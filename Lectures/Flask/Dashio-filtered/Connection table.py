# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 13:59:09 2021

@author: ankit
"""

import pyodbc
import psycopg2
import pandas as pd



cndBase = pyodbc.connect('''DRIVER={MySQL ODBC 8.0 ANSI Driver}; SERVER=localhost;
                         DATABASE=ninjatrader; UID=admin; PASSWORD=1234;''',autocommit=True) 
sql = """
CREATE TABLE ninjatrader.accounts (
                username text NOT NULL,
                password text NOT NULL,
                id text NOT NULL,
                address text NOT NULL,
                city text NOT NULL,
                country text NOT NULL,
                postalcode bigint NOT NULL,
                aboutme text NOT NULL)
"""

pd.read_sql(sql, con=cndBase)

sql = """
select * from ninjatrader.accounts 
"""

pd.read_sql(sql, con=cndBase)
    