import numpy as np
import os
os.chdir('F:\LocalDriveD\Teaching')
######## Saved Objects ########
import pickle
filenamemodel = 'finalized_model.pkl'
# load the model from disk
with open(filenamemodel, 'rb') as file:  
    clf = pickle.load(file)


from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

@app.route('/')
def student():
   return render_template('HotelCustomer.html')

## render_template s used to display output at browser
@app.route('/prediction',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      new_value = np.array([[result['adr'],result['children']]])
      result2 = {'Name of Customer':result['Name'],'Average Daily Rate':result['adr'],
                 'No_of_Children':result['children'],
                 'Prediction':clf.predict(new_value)[0]}
      return render_template("prediction.html",result = result2)
     
if __name__ == '__main__':
   app.run(debug = True,use_reloader=False)
   
   
   
   
   
   
   '''
### To get single prediction using post request
import json
import pandas as pd
@app.route('/prediction',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      new_value = np.array([[result['adr'],result['children']]])
      data = pd.DataFrame({
          'Name of Customer':[result['Name']],
          'Average Daily Rate':[result['adr']],
          'No_of_Children':[result['children']],
          'Prediction':[clf.predict(new_value)[0]]})
      return  data.to_json()
'''
