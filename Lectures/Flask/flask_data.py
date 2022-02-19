from flask import Flask, render_template, request
app = Flask(__name__)
import os

@app.route('/')
def student():
   return render_template('employee.html')

@app.route('/Age',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      return os.system("start /wait cmd /c python F:\\LocalDriveD\\Analytics\\Freelancing\\Upwork\\Algo_Trading\\practice\\dash_practice9.py")

if __name__ == '__main__':
   app.run(debug=True, use_reloader=False,port='5000')