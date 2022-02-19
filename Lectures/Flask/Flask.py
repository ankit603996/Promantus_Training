from flask import Flask
from flask import redirect, url_for
app = Flask(__name__)
#import sys
#new_path = 'F:\LocalDriveD\Teaching'
#if new_path not in sys.path:
#    sys.path.append(new_path)
#import myscript
#app.debug=True

@app.route('/') #The route() function of the Flask class is a decorator, which tells the application which URL should call the associated function
def function1():
   #return myscript.my_function2()
    
    return "Show Flask Demo"

@app.route('/<variable>') 
def function2(variable):
    return 'Show Flask Demo for %s!' % variable

@app.route('/<int:EmployeeNumber>')
def function3(EmployeeNumber):
   return 'Show Flask Demo for employee %d' % EmployeeNumber

@app.route('/checkfunction4/<employee>')
def function4(employee):
    if employee == 'ankit':
        return redirect(url_for('function2',variable = employee))
    else:
        return redirect(url_for('function3',EmployeeNumber=employee))


if __name__ == '__main__': #code to be executed when this file is run directly, but not when it is imported by another module.
   app.run(debug=False, use_reloader=True) #https://stackoverflow.com/questions/49456385/running-flask-from-ipython-raises-systemexit

