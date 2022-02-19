# Import Libraries required for the experiment
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import QuantileTransformer,MinMaxScaler,RobustScaler,StandardScaler
import random
import os
import seaborn as sns
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeClassifier ,DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
from sklearn import tree
import collections
np.random.seed(132)
# Initiate Seaborn Package with a particular Style
sns.set(style="ticks", color_codes=True)

# Initiate Encoders which will be used for Categorical Variables
label_encoder = LabelEncoder()

# Prints all line without skipping
pd.set_option('display.max_rows', None)

# load the raw input dataset
bookings=pd.read_csv('D:\\Proj\\Coaching\\Coaching-ML\\ML Algorithms\\Decision Tree\\hotel-booking-demand\\hotel_bookings.csv')
bookings.head(10)
print(bookings.columns.values)

#Assign Temporary Storage of the core dataset (For larger sets this step must be avoided)
temp=bookings.sample(10000)
temp=temp.drop(temp.columns[[5,6,13,23,24,31]],axis=1)
# Arrive at the logical columns that can be Categoric
# Conversion of String values to NaN if the real value is supposed to be Numeric
#temp.iloc[:,[0,1,3,13]]=temp.iloc[:,[0,1,3,13]].apply(lambda s: pd.to_numeric(s, errors='coerce'))
#print (temp.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()))

# Column Null Value Check
temp['is_canceled'].isnull()

# Retrieve Columns that have Null Value 
null_columns=temp.columns[temp.isnull().any()]

# Retrieve Sample Rows for columns that have Null Value 
print(temp[temp.isnull().any(axis=1)][null_columns].head())

# Function to Point Out the Columns & Rows that have null values 
for i in range(1,len(null_columns)):
    print(null_columns[i])
    print('____________________________________________________________')
    print((pd.notnull(temp)[null_columns[i]]).value_counts())
    print(temp[(pd.isnull(temp)[null_columns[i]])][null_columns[i]])
    print('============================================================')

# Identify Categorical Variables for Better Automation of the entire script
temp1=temp.head()
temp1.iloc[:,[0,1,3,4,10,11,12,13,16,17,19,21]] = temp1.iloc[:,[0,1,3,4,10,11,12,13,16,17,19,21]].astype(str)
temp1.iloc[:,[0,1,3,4,10,11,12,13,16,17,19,21]].apply(LabelEncoder().fit_transform)


categoric_variables = temp1.columns[temp1.dtypes=='object']

# Categorical Imputation
for i in range(0,len(categoric_variables)):
    print(categoric_variables[i])
    print(temp[categoric_variables[i]].mode()[0])
    temp[categoric_variables[i]]=temp[categoric_variables[i]].replace(np.nan,temp[categoric_variables[i]].mode()[0])
    temp[categoric_variables[i]]=temp[categoric_variables[i]].replace(to_replace ="NaN", 
                 value =0)
    temp[categoric_variables[i]].fillna(0, inplace = True) 

# Numeric Imputation
from sklearn.impute import SimpleImputer
mean_imp = SimpleImputer(missing_values=np.nan, strategy="mean")
med_imp = SimpleImputer(missing_values=np.nan, strategy="median")

# Mean Imputation for Numeric Variables
numeric_variables = set(temp.columns) - set(categoric_variables)
i=1
for i in range(0,len(numeric_variables)):
    if(temp.columns[i] in numeric_variables):
       print(temp.columns[i])        
       temp.iloc[:,[i]] = mean_imp.fit_transform(temp.iloc[:,[i]])
       temp[temp.columns[i]]=temp[temp.columns[i]].replace(to_replace ="NaN", 
                 value =0)
       temp[temp.columns[i]].fillna(0, inplace = True) 

# Making Categoric for String Variables   
temp['hotel'] = temp['hotel'].astype(str)
temp['arrival_date_year'] = temp['arrival_date_year'].astype(str)
temp['arrival_date_month'] = temp['arrival_date_month'].astype(str)
temp['meal'] = temp['meal'].astype(str)
temp['market_segment'] = temp['market_segment'].astype(str)
temp['distribution_channel'] = temp['distribution_channel'].astype(str)
   
temp['is_repeated_guest'] = temp['is_repeated_guest'].astype(str)
temp['reserved_room_type'] = temp['reserved_room_type'].astype(str)
temp['assigned_room_type'] = temp['assigned_room_type'].astype(str)
temp['deposit_type'] = temp['deposit_type'].astype(str)
temp['customer_type'] = temp['customer_type'].astype(str)
temp['reservation_status'] = temp['reservation_status'].astype(str)

temp['is_canceled'] = temp['is_canceled'].astype(str)

# Plotting All the Variables that are available in raw file
os.chdir('D:\\Proj\\Coaching\\Coaching-ML\\ML Algorithms\\Decision Tree\\')
if not os.path.exists('Plots'):
    os.makedirs('Plots')
os.chdir('D:\\Proj\\Coaching\\Coaching-ML\\ML Algorithms\\Decision Tree\\Plots\\')

for i in range(1,len(temp.columns)):
    if(temp.dtypes[i] in ('float64','int64')):
        print(temp.columns[i])
        plt.figure()
        sns_plot = sns.distplot(temp[temp.columns[i]], kde=False)
        sns_plot.figure.savefig(temp.columns[i]+'.png')
    else:
        plt.figure()
        sns_plot = sns.catplot(x=temp.columns[i], kind="count", palette="ch:.25", data=temp)
        sns_plot.savefig(temp.columns[i]+'.png')


# Log transformation with mean Imputation for Non-Uniform Variables
log_mean_transform_var=['adr','lead_time','stays_in_week_nights']

# Plotting All the Transformed Variables 
os.chdir('D:\\Proj\\Coaching\\Coaching-ML\\ML Algorithms\\Decision Tree\\')
if not os.path.exists('Transformed Plots'):
    os.makedirs('Transformed Plots')
os.chdir('D:\\Proj\\Coaching\\Coaching-ML\\ML Algorithms\\Decision Tree\\Transformed Plots\\')

import time
for i in range(0,len(temp.columns)):
    if(temp.columns[i] in log_mean_transform_var):
        logtransform=np.log(temp[temp.columns[i]])
        logtransform = logtransform.replace([np.inf, -np.inf], np.nan)
        logtransform = mean_imp.fit_transform(pd.DataFrame(logtransform))
        temp[temp.columns[i]]=logtransform
        temp[temp.columns[i]].fillna(0, inplace = True) 

for i in range(1,len(temp.columns)):
    if(temp.columns[i] in log_mean_transform_var):
        plt.figure()
        sns_plot = sns.distplot(temp[temp.columns[i]], kde=False)
        sns_plot.figure.savefig(temp.columns[i]+'.png')

# Create correlation matrix
corr_matrix = temp[numeric_variables].corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.7 & less than -0.7
to_drop_pos = [column for column in upper.columns if any(upper[column] > 0.7)]
to_drop_neg = [column for column in upper.columns if any(upper[column] < -0.7)]

plt.figure(figsize=(10,10))
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=np.bool))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns_plot=sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns_plot.figure.savefig('correlation.png')
# Fill na values with Numeric value as 0
for i in range(1,len(temp.columns)):
    if(temp.dtypes[i] in ('float64','int64')):
        temp[temp.columns[i]].fillna(0, inplace = True) 


# Drop NaN Value from the temp variable
        
temp=temp.dropna(how='any') 


X = temp.drop(temp.columns[1],axis=1)
y = temp[temp.columns[1]]


# Identify Numeric Values separately
num_only=set(numeric_variables).difference(categoric_variables)


# Feature Importance for numeric variables 
# fit an Extra Trees model to the data
et_model_num = ExtraTreesClassifier()
et_model_num.fit(X[num_only], y)
# display the relative importance of each attribute
print(et_model_num.feature_importances_)
et_p_values_num = pd.Series(et_model_num.feature_importances_,index = X[num_only].columns)
et_p_values_num.sort_values(ascending = False , inplace = True)
et_p_values_num.index
plt.figure()
sns.set(rc={'figure.figsize':(11.7,8.27)})
pal = sns.color_palette("Greens_d")
sns.set_context("paper")
sns_plot = sns.barplot(et_p_values_num.index[et_p_values_num>0],et_p_values_num[et_p_values_num>0],alpha = 0.85)
plt.xticks(rotation=90)
plt.xlabel('Factors', fontsize = 11, weight = 'bold')
plt.ylabel('Feature Importance VALUE', fontsize = 11, weight = 'bold')
sns_plot.set_title("Variable Selection using Feature Importance", fontsize = 20, weight = 'bold')
sns_plot.figure.savefig('Feature_Importance'+'.png')
et_selected_num=et_p_values_num.index[et_p_values_num>0]
et_rejected_num=set(X[num_only].columns.values).difference(et_selected_num)
#GINI INDEX for numeric variables 

# Decision tree with gini 
dt_model_num = DecisionTreeClassifier( criterion='gini',
             random_state = 100, 
            max_depth = 5, min_samples_leaf = 20) 
  
# Performing training 
dt_model_num.fit(X[num_only], y)
print(dt_model_num.feature_importances_)

dt_p_values_num = pd.Series(dt_model_num.feature_importances_,index = X[num_only].columns)
dt_p_values_num.sort_values(ascending = False , inplace = True)
dt_p_values_num.index
plt.figure()
sns.set(rc={'figure.figsize':(11.7,8.27)})
pal = sns.color_palette("Greens_d")
sns.set_context("paper")
sns_plot = sns.barplot(dt_p_values_num.index[dt_p_values_num>0],dt_p_values_num[dt_p_values_num>0],alpha = 0.85)
plt.xticks(rotation=90)
plt.xlabel('Factors', fontsize = 11, weight = 'bold')
plt.ylabel('GINI INDEX VALUE', fontsize = 11, weight = 'bold')
sns_plot.set_title("Variable Selection using Gini Scores", fontsize = 20, weight = 'bold')
sns_plot.figure.savefig('GINI_INDEX'+'.png')

dt_selected_num=dt_p_values_num.index[dt_p_values_num>0]
dt_rejected_num=set(X[num_only].columns.values).difference(dt_selected_num)

temp.iloc[:,[0,1,3,4,10,11,12,13,16,17,19,21]] = temp.iloc[:,[0,1,3,4,10,11,12,13,16,17,19,21]].astype('category')
temp.iloc[:,[0,1,3,4,10,11,12,13,16,17,19,21]].apply(LabelEncoder().fit_transform)

# Selecting Categorical Variables alone for Variable Selection
t=temp.iloc[:,[0,1,3,4,10,11,12,13,16,17,19,21]]
# Dropping off NaN Values if Any
t=t.dropna(how='any')
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for i in range(0,len(t.columns)):
    t.iloc[:,i] = le.fit_transform(t.iloc[:,i])
# Using GINI to decide Variables for Categoric Variables
from sklearn.tree import DecisionTreeRegressor 

dt_model_cat = DecisionTreeClassifier( criterion='gini',
             random_state = 100, 
            max_depth = 10, min_samples_leaf = 10)
dt_model_cat.fit(t, y) 
print(dt_model_cat.feature_importances_)

# dt_p_values_cat = pd.Series(dt_model_cat.feature_importances_,index = t.columns)
# dt_p_values_cat.sort_values(ascending = False , inplace = True)
# dt_p_values_cat.index
# plt.figure()
# sns.set(rc={'figure.figsize':(11.7,8.27)})
# pal = sns.color_palette("Greens_d")
# sns.set_context("paper")
# sns_plot = sns.barplot(dt_p_values_cat.index[dt_p_values_cat>0],dt_p_values_cat[dt_p_values_cat>0],alpha = 0.85)
# plt.xticks(rotation=90)
# plt.xlabel('Factors', fontsize = 15, weight = 'bold')
# plt.ylabel('GINI INDEX VALUE', fontsize = 15, weight = 'bold')
# sns_plot.set_title("Variable Selection using Gini Scores (Categoric)", fontsize = 20, weight = 'bold')
# sns_plot.figure.savefig('GINI_INDEX_Categoric'+'.png')

# dt_selected_cat=dt_p_values_cat.index[dt_p_values_cat>0]
# dt_rejected_cat=dt_p_values_cat.index[dt_p_values_cat==0]


# Feature Importance for categoric variables 
# fit an Extra Trees model to the data
et_model_cat = ExtraTreesClassifier()
et_model_cat.fit(t, y) 
# display the relative importance of each attribute
print(et_model_cat.feature_importances_)
et_p_values_cat = pd.Series(et_model_cat.feature_importances_,index = t.columns)
et_p_values_cat.sort_values(ascending = False , inplace = True)
et_p_values_cat.index
plt.figure()
sns.set(rc={'figure.figsize':(11.7,8.27)})
pal = sns.color_palette("Greens_d")
sns.set_context("paper")
sns_plot = sns.barplot(et_p_values_cat.index[et_p_values_cat>0],et_p_values_cat[et_p_values_cat>0],alpha = 0.85)
plt.xticks(rotation=90)
plt.xlabel('Factors', fontsize = 11, weight = 'bold')
plt.ylabel('Feature Importance VALUE', fontsize = 11, weight = 'bold')
sns_plot.set_title("Variable Selection using Feature Importance", fontsize = 20, weight = 'bold')
sns_plot.figure.savefig('Feature_Importance'+'.png')
et_selected_cat=et_p_values_cat.index[et_p_values_cat>0.002]
et_rejected_cat=set(t.columns.values).difference(et_selected_cat)

rejected_variables=[]
rejected_variables.append(to_drop_pos)
rejected_variables.append(to_drop_neg)
rejected_variables.append(dt_rejected_num)
#rejected_variables.append(dt_rejected_cat)
rejected_variables.append(et_rejected_cat)
rejected_variables.append(et_rejected_num)

flattened_list = [y for x in rejected_variables for y in x]
rejected_variables_filter = list(dict.fromkeys(flattened_list))
print(rejected_variables_filter)


print(set(temp.columns).difference(rejected_variables_filter))

final_selection=set(temp.columns).difference(rejected_variables_filter)
final_selection.remove('is_canceled')

cat_vars=set(set(final_selection).difference(rejected_variables_filter)).difference(numeric_variables)
cat_vars1=cat_vars
final_selection=set(final_selection).difference(cat_vars1)
cat_vars.remove('reservation_status')
cat_vars.remove('assigned_room_type')
cat_vars.remove('hotel')
cat_vars.remove('distribution_channel')



data=pd.DataFrame()
data1=pd.DataFrame()
#cat_vars=['reservation_status','customer_type','market_segment','reserved_room_type','hotel','distribution_channel','arrival_date_year','assigned_room_type','meal','deposit_type','arrival_date_month',]
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(X[var], prefix=var)
    data1=pd.concat([data1,cat_list] ,axis=1)
    #data=data1

# Model Initiation

X_log=X[final_selection]
X_log=pd.concat([X_log,data1], axis=1)

RANDOM_SEED = 30
X_train, X_test, y_train, y_test = train_test_split(X_log, y, test_size=0.33, random_state=RANDOM_SEED)
# Number of features to consider at every split
max_features = ['auto', 'sqrt','log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(3, 50, num = 30)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [5,10,15,20,25,30,35,40]
# Minimum number of samples required at each leaf node
min_samples_leaf = [5,7,10,15,20]
# Maximum number of samples required at each leaf node
max_leaf_nodes = [5,7,10,15,20]
# Splitting of the tree based on gini or entropy
criterion =['gini','entropy']
# Splitting of the tree 
splitter=['best','random']
min_weight_fraction_leaf=[0.0,0.1,0.2,0.3]
min_impurity_decrease=[0.0,0.1,0.2,0.3]
random_stat=[5,10,15]
# Create the random grid
random_grid = {'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'max_leaf_nodes': max_leaf_nodes,
               'criterion': criterion,
               'splitter':splitter,
               #'min_weight_fraction_leaf':min_weight_fraction_leaf,
               'min_impurity_decrease':min_impurity_decrease
               }
print(random_grid)
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(
 class_weight=None,
 presort=False,random_state=False)
print(clf.get_params())
dt_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
dt_random.fit(X_train,y_train)
best_random_dt = dt_random.best_estimator_
#Predict the response for train dataset
y_pred_train_randome = best_random_dt.predict(X_train)
#Predict the response for test dataset
y_pred_test_randome = best_random_dt.predict(X_test)

# Train Data set metrics
pd.DataFrame(
    confusion_matrix(y_train, y_pred_train_randome),
    columns=['Predicted Not Cancelled', 'Predicted Cancelled'],
    index=['True Not Cancelled', 'True Cancelled']
)

accuracy_score(y_train, y_pred_train_randome)
classification_report(y_train, y_pred_train_randome)

# test Data set metrics
pd.DataFrame(
    confusion_matrix(y_test, y_pred_test_randome),
    columns=['Predicted Not Cancelled', 'Predicted Cancelled'],
    index=['True Not Cancelled', 'True Cancelled']
)

accuracy_score(y_test, y_pred_test_randome)
classification_report(y_test, y_pred_test_randome)



# Plotting the Decision Tree
dot_data = StringIO()
indep_var=X_train.columns.values
features = [u'{}'.format(c) for c in indep_var]
export_graphviz(best_random_dt, out_file=dot_data, 
                feature_names=features,
                filled=True, rounded=True,
                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
graph.write_png("D:\\Proj\\Coaching\\Coaching-ML\\ML Algorithms\\Decision Tree\\dtree4.png")

# Custom Colorization


colors = ('green','orange')
edges = collections.defaultdict(list)

for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))

for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])

graph.write_png("D:\\Proj\\Coaching\\Coaching-ML\\ML Algorithms\\Decision Tree\\dtree5.png")

