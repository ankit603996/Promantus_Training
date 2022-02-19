###############Public###################
### API to fetch data using URL
import requests
response = requests.get("https://open.er-api.com/v6/latest/USD")
print(response.status_code)
print(response.json())
print(response.json()['rates'])

import requests
response = requests.get("https://datausa.io/api/data?drilldowns=Nation&measures=Population")
print(response.status_code)
print(response.json())
print(response.json()['data'])
print(response.json()['data'][1])
print(response.json()['data'][1]['ID Nation'])





















###############################################################################
###############################################################################
###############################################################################



###############flask_basic_api.py###################
### API to fetch data using URL
import requests
response = requests.get("http://127.0.0.1:5000//employeedata")
print(response.status_code)
print(response.json())

### API to fetch single data using URL
import requests
response = requests.get("http://127.0.0.1:5000//employeedata/1")
print(response.status_code)
print(response.json())

### API to fetch single data using query parameters
import requests
response = requests.get("http://127.0.0.1:5000//employeedatanew/1") # will show 404
print(response.status_code)
response = requests.get("http://127.0.0.1:5000//employeedatanew?id=1") 
print(response.status_code)
print(response.json())






###############################################################################
###############################################################################
###############################################################################

###############flask_data.py###################
import requests
response = requests.get("http://127.0.0.1:5000//employeedetails")
print(response.status_code)
print(response.json())

URL = "http://127.0.0.1:5000//employeedetails"

import requests
response = requests.get(url = URL)
print(response.status_code)
print(response.json())


#requests.post method could be used for many other tasks as well like filling and submitting the web forms
#we will need to pass some data to API server. We store this data as a dictionary.
#server processes the data sent to it and sends the  URL of your source_code which can be simply accessed by r.text .
# in this case we are using webapi 
data = {'Marketing':64,
        'Sales':65,
        'Admin':59,
        'Name':'Ankit'}
import requests
response = requests.post("http://127.0.0.1:5000//Age",data = data)
print(response.status_code)
print(response.text)
###############################################################################
###############################################################################
###############################################################################





#consume the service through its REST interface
###############flask_decision_tree.py###################
data = {'Name':'Ankit',
        'adr':65,
        'children':0}
import requests
response = requests.post("http://127.0.0.1:5000/prediction",data = data)
print(response.status_code)
print(response.text)
print(response.json())
print(response.json()['Prediction']['0'])



