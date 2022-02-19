# Import Libraries required for the experiment
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer,MinMaxScaler,RobustScaler,StandardScaler
import random
import os
import seaborn as sns
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier ,DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report 
#from sklearn.tree import export_graphviz
#import pydotplus
from sklearn import tree
import collections
np.random.seed(132)


os.getcwd()


# ### Data Snapshot

# load the raw input dataset
os.chdir('F:\LocalDriveD\Teaching')
bookings=pd.read_csv('hotel_bookings.csv')
bookings.head(10)


# In[5]:


bookings.info()


bookings.nunique()


temp = bookings[['adr','children','is_canceled']]
temp.info()
temp = temp.dropna()
temp.info()
y = temp['is_canceled']
X_log = temp[['adr','children']]

X_log.head()


# ### Decision Tree1

RANDOM_SEED = 30
X_train, X_test, y_train, y_test = train_test_split(X_log, y, test_size=0.33, random_state=RANDOM_SEED)
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(max_depth=5)
# Fit the random search model
clf.fit(X_train,y_train)

#Predict the response for train dataset
y_pred_train = clf.predict(X_train)
#Predict the response for test dataset
y_pred_test= clf.predict(X_test)

confusion_matrix_train = confusion_matrix(y_train, y_pred_train)

print( "Classification report train data\n\n" ,classification_report(y_train, y_pred_train))

print( "Classification report train data\n\n" ,classification_report(y_test, y_pred_test))

################################# Prediction on new values ############

new_value = np.array([[79.0,0.0]])
clf.predict(new_value)
new_value2 = np.array([[62.0,0.0]])
clf.predict(new_value2)


################################# store model object ############
######## Save Objects ########
import pickle
# save the model to disk
filenamemodel = "finalized_model.pkl"  
with open(filenamemodel, 'wb') as file:  
    pickle.dump(clf, file)

