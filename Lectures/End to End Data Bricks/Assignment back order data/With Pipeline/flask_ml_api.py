import numpy as np
import os

def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
    url = 'https://adb-2521914531602271.11.azuredatabricks.net/model/AG_DT_Pipeine/2/invocations'
    headers = {'Authorization': f'Bearer dapibf0d31f9a029172aac7ae47aa7933881-3'}
    data_json = dataset.to_dict(orient='split') if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    response = requests.request(method='POST', headers=headers, url=url, json=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')
    return response.json()


from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

@app.route('/')
def student():
   return render_template('Back_Order.html')

## render_template s used to display output at browser
@app.route('/prediction',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      new_value = pd.DataFrame({
      'national_inv': result['national_inv'],
     'lead_time': result['lead_time'],
     'in_transit_qty': result['in_transit_qty'],
     'forecast_3_month': result['forecast_3_month'],
     'forecast_6_month': result['forecast_6_month'],
     'forecast_9_month': result['forecast_9_month'],
     'sales_1_month': result['sales_1_month'],
     'sales_3_month': result['sales_3_month'],
     'sales_6_month': result['sales_6_month'],
     'sales_9_month': result['sales_9_month'],
     'min_bank': result['min_bank'],
     'pieces_past_due':result['pieces_past_due'],
     'perf_12_month_avg': result['perf_12_month_avg'],
     'perf_6_month_avg': result['perf_6_month_avg'],
     'local_bo_qty': result['local_bo_qty'],
     'potential_issue': result['potential_issue'],
     'deck_risk': result['deck_risk'],
     'ppap_risk': result['ppap_risk'],
     'stop_auto_buy': result['stop_auto_buy'],
     'oe_constraint': result['oe_constraint'],
     'rev_stop': result['rev_stop']},index=[0])
      result2 = {'Prediction':score_model(new_value)}
      return render_template("prediction.html",result = result2)
     
if __name__ == '__main__':
   app.run(debug = True,use_reloader=False)
   
   
   
   
   
