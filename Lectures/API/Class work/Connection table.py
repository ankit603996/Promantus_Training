import pyodbc
#import psycopg2
import pandas as pd



cndBase = pyodbc.connect('''DRIVER={MySQL ODBC 8.0 ANSI Driver}; SERVER=localhost;
                         DATABASE=ninjatrader; UID=admin; PASSWORD=1234;''',autocommit=True) 
sql = """
CREATE TABLE ninjatrader.employee (
                id bigint NOT NULL,
                employeename text NOT NULL,
                department text NOT NULL,
                address text NOT NULL,
                rating text NOT NULL)
"""

pd.read_sql(sql, con=cndBase)

sql = """
select * from ninjatrader.employee 
"""
pd.read_sql(sql, con=cndBase)

employee = pd.read_excel(r'F:\LocalDriveD\Teaching\employee.xlsx')

cursor = cndBase.cursor()
# Insert Dataframe into SQL Server:
for index, row in employee.iterrows():
            cursor.execute("INSERT INTO employee (id,employeename,department,address,rating) values(?,?,?,?,?)", row.id, row.employeename, row.department,row.address,row.rating)
cndBase.commit()
cursor.close()

sql = """
select * from ninjatrader.employee 
"""
pd.read_sql(sql, con=cndBase)

