# Feature Importance
from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
hotel=pd.read_csv('D://Proj//Coaching//Coaching-ML//Feature Engineering//hotel-booking-demand//hotel_bookings.csv')
hotel.head(10)
hotel=hotel.head(100)
temp=hotel.iloc[:,[0,1,2,7,8,9,10,11,12,13,14,15,16]]
label_encoder = LabelEncoder()
temp['hotel'] = label_encoder.fit_transform(temp['hotel'])
temp['meal'] = label_encoder.fit_transform(temp['meal'])

temp['country'] = temp['country'].astype(str)
temp['country'] = label_encoder.fit_transform(temp['country'])
temp['distribution_channel'] = temp['distribution_channel'].astype(str)
temp['distribution_channel'] = label_encoder.fit_transform(temp['distribution_channel'])
temp['market_segment'] = temp['market_segment'].astype(str)
temp['market_segment'] = label_encoder.fit_transform(temp['market_segment'])
temp = temp.drop('is_canceled',axis=1)

ht=pd.get_dummies(temp, columns=['hotel'])
ml=pd.get_dummies(temp, columns=['meal'])
co=pd.get_dummies(temp, columns=['country'])
dc=pd.get_dummies(temp, columns=['distribution_channel'])
ms=pd.get_dummies(temp, columns=['market_segment'])

dataset=temp
dataset = dataset.drop('hotel',axis=1)
dataset = dataset.drop('meal',axis=1)
dataset = dataset.drop('country',axis=1)
dataset = dataset.drop('distribution_channel',axis=1)
dataset = dataset.drop('market_segment',axis=1)

dataset=pd.concat([dataset,ht],axis=1)
dataset=pd.concat([dataset,ml],axis=1)
dataset=pd.concat([dataset,dc],axis=1)
dataset=pd.concat([dataset,ms],axis=1)
#dataset=pd.concat([dataset,ht],axis=1)

# Numeric Imputation
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
imp1 = SimpleImputer(missing_values=np.nan, strategy="median")
X = imp.fit_transform(dataset)


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
pca = PCA(n_components=3)

pca.fit(X)
X1=pca.fit_transform(X)

var= pca.explained_variance_ratio_

#Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

plt.plot(var1)

u, s, v = np.linalg.svd(X, full_matrices=True)

from sklearn.decomposition import TruncatedSVD


svd_tr = TruncatedSVD(n_components=20)
X3=svd_tr.fit(X)

tsvd=svd_tr.components_.round(2)