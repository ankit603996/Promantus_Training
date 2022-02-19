from flask import Flask
from flask import render_template

app = Flask(__name__)

#@app.route('/')
#def function1():
#   return '<html><body><h2>Show Flask Demo</h2></body></html>'

#@app.route('/')
#def function1():
#   return render_template('show_demo1.html')

@app.route('/')
def function1():
    listofemployee = {'employe1':101,'employe2':102,'employe3':103}
    my_name= 'my name ankit'
    my_name2 = 'my name ankit 2' 
    return render_template('employeelist.html',employeelist = listofemployee,
                           name = my_name, name2 = my_name2) # i can pass more than one variables


if __name__ == '__main__':
   app.run(debug = True)
   

















'''
@app.route('/')
def function1():
   return render_template('show_demo1.html')

'''



'''
@app.route('/')
def function1():
    listofemployee = {'employe1':101,'employe2':102,'employe3':103}
    my_name= 'my name xyz'
    return render_template('employeelist.html',employeelist = listofemployee,
                           name = my_name) # i can pass more than one variables
'''