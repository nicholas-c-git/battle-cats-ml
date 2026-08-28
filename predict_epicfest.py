import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

#encoding series id to either 1 for epicfest, or 0
from sklearn.preprocessing import OneHotEncoder

#basic classification models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data_important = pd.read_csv('series.csv', parse_dates=[0,1],date_format='%Y-%m-%d')

#initialize a one hot encoder in pandas DataFrame format
one_hot = OneHotEncoder(sparse_output=False).set_output(transform='pandas')

#make a DataFrame of the series id column but encoded
#series id 27 is epicfest
data_encoded = one_hot.fit_transform(data_important[['series id']])

#making a dataframe of the features to be used
data_compiled = pd.concat(
    [data_important[['start date', 'end date']],
     data_encoded[['series id_27']]],
    axis=1) #columns

print(data_compiled)
#probably going to transform the 'start date' (YYYY-MM-DD) into just month and day or
#time since start of year.
#also want to turn 'end date' into a duration instead (start date - end date)