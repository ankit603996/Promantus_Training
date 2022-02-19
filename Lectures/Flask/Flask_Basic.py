from flask import Flask

#WSGI Application
app = Flask(__name__) #Flask constructor takes the name of current module (__name__) as argument

@app.route('/') #The route() function of the Flask class is a decorator, 
#which tells the application which URL should call the associated function
def function1():
    return "Show Flask Demo"

@app.route('/about-us') 
def function2():
    return "about-us page"

@app.route('/contact-us') 
def function3(variable):
    
    return "contact us at ankit@datadna.in" 

#code to be executed when this file is run directly,
#but not when it is imported by another module.
if __name__ == '__main__': 
   app.run(host = '127.0.0.1',port ='5000',debug=True, use_reloader=True) 

