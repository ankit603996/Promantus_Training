from flask import Flask
from flask import redirect, url_for

app = Flask(__name__)

@app.route('/') #The route() function of the Flask class is a decorator, which tells the application which URL should call the associated function
def function1():
    return "Show Flask Demo"

@app.route('/<variable1>') 
def function2(variable1):
    return 'Flask Demo: Show employee name %s!' % variable1

@app.route('/<int:EmployeeNumber>')
def function3(EmployeeNumber):
   return 'Flask Demo: Show employee ID %d' % EmployeeNumber

@app.route('/checkfunction4/<employee>')
def function4(employee):
    if type(employee) == str:
        return redirect(url_for('function2',variable1 = employee))
    else:
        return redirect(url_for('function3',EmployeeNumber=employee))
    

if __name__ == '__main__': #code to be executed when this file is run directly, but not when it is imported by another module.
   app.run(debug=True, use_reloader=True) 




'''

@app.route('/checkfunction4/<employee>')
def function4(employee):
    if type(employee) == str:
        return redirect(url_for('function2',variable = employee))
    else:
        return redirect(url_for('function3',EmployeeNumber=employee))


'''
