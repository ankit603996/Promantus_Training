import flask
from flask import request, jsonify

app = flask.Flask(__name__)

# Create some test data for our employees
employee = [
    {'id': 0,
     'employeename': 'John',
     'department': 'marketing'},
    {'id': 1,
     'employeename': 'Leena',
     'department': 'Sales'},
    {'id': 2,
     'employeename': 'Haris',
     'department': 'Sales'}
]


@app.route('/', methods=['GET'])
def index():
    return '''Basic API for Employee data Fatchs'''


# URL for our API where user can get all employee information
@app.route('/employeedata', methods=['GET'])
def api_all():
    return jsonify(employee)




app.run(debug=False)




'''
# URL for our API where user can get any single employee information
@app.route('/employeedata/<int:employee_id>', methods=['GET'])
def api_one(employee_id):
    result = []
    for emp in employee:
        if emp['id'] == employee_id:
            result.append(emp)
    return jsonify(result)

'''
'''
# URL for our API where user can get any single employee information
#using query parameters (which are used in url after '?')
@app.route('/employeedatanew', methods=['GET'])
def api_one():
    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.
    if 'id' in request.args:
        id = int(request.args['id'])
    else:
        return "Error: No id field provided. Please specify an id."

    # Create an empty list for our results
    results = []

    # Loop through the data and match results that fit the requested ID.
    # IDs are unique, but other fields might return many results
    for emp in employee:
        if emp['id'] == id:
            results.append(emp)
    # Use the jsonify function from Flask to convert our list of
    # Python dictionaries to the JSON format.
    return jsonify(results)'''
