import numpy as np
import os
os.chdir('F:\LocalDriveD\Teaching')
######## Saved Objects ########
import pickle
filenamemodel = 'finalized_model.pkl'
# load the model from disk
with open(filenamemodel, 'rb') as file:  
    clf = pickle.load(file)

new_value = np.array([[79.0,0.0]])
clf.predict(new_value)
new_value2 = np.array([[62.0,0.0]])
clf.predict(new_value2)

