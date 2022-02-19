from flask import Flask, render_template, request, jsonify
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re
import pandas as pd
import pyodbc

app = Flask(__name__)


# Enter your database connection details below
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'admin'
app.config['MYSQL_PASSWORD'] = '1234'
app.config['MYSQL_DB'] = 'ninjatrader'

# Intialize MySQL
mysql = MySQL(app)



@app.route('/')
def student():
   return render_template('employee.html')



@app.route('/employeedetails',methods = ['GET']) # by default method is get
def employee():
   cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
   cursor.execute('SELECT * FROM employee')
   employeedata = cursor.fetchall()
   return jsonify(employeedata)

@app.route('/employeedetails/<int:employee_id>',methods = ['GET'])
def employee2(employee_id):
   cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
   cursor.execute('SELECT * FROM employee where id = %s', (employee_id,))
#   cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
   employeedata = cursor.fetchall()
   return jsonify(employeedata)


@app.route('/Age',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      return render_template("Age.html",result = result)

if __name__ == '__main__':
   app.run(debug = True,use_reloader=False)