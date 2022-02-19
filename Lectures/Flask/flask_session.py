from flask import Flask, session, redirect, url_for, escape, request,render_template
app = Flask(__name__)
from datetime import timedelta
app.secret_key = '1234567890qwertyuiop'
app.permanent_session_lifetime = timedelta(minutes = 5) # store data for 5 minutes and delete after that

#URL ‘/’ simply prompts user to log in, as session variable ‘username’ is not set.
#If the user name doesn’t exist then redirect to the login page.
@app.route('/')
def index():
   if 'username' in session:
      username = session['username']
      return 'Logged in as ' + username + '<br>' + \
             "<b><a href = '/logout'>click here to log out</a></b>"
   return "You are not logged in <br><a href = '/login'></b>" + \
      "click here to log in</b></a>"

#A Form is posted back to ‘/login’ and now session variable is set. 
#After storing user information application is redirected to ‘/’. 
#This time session variable ‘username’ is found.
@app.route('/login', methods = ['GET', 'POST'])
def login():
   if request.method == 'POST':
       session.permanent = True
       session['username'] = request.form['username']
       return redirect(url_for('index'))
   return render_template('login.html')

#The application also contains a logout() view function, which pops out ‘username’ session variable.
# Hence, ‘/’ URL again shows the opening page.   
@app.route('/logout')
def logout():
   # remove the username from the session if it is there
   session.pop('username', None)
   return redirect(url_for('index'))

if __name__ == '__main__':
   app.run(debug = True,use_reloader=True)


'''
make session permanent add below lines

from datetime import timedelta
app.secret_key = '1234567890qwertyuiop'
app.permanent_session_lifetime = timedelta(minutes = 5) # store data for 5 minutes and delete after that


@app.route('/login', methods = ['GET', 'POST'])
def login():
   if request.method == 'POST':
       session.permanent = True
       session['username'] = request.form['username']
       return redirect(url_for('index'))
   return render_template('login.html')



'''