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

data_important = pd.read_csv('series.csv')

#date columns as a datetime
start_date = pd.to_datetime(data_important['start date'])
end_date = pd.to_datetime(data_important['end date'])

#initialize a one hot encoder in pandas DataFrame format
one_hot = OneHotEncoder(sparse_output=False).set_output(transform='pandas')

#make a DataFrame of the series id column but encoded
#convert_dtypes because the 1s and 0s were 1.0 and 0.0
data_encoded = one_hot.fit_transform(data_important[['series id']]).convert_dtypes()

#series id 27 is epicfest
data_27 = data_encoded['series id_27']

#making a dataframe of the features to be used
data_compiled = pd.concat(
    [start_date.dt.month,
     start_date.dt.day,
     end_date - start_date, #duration in days
     start_date - pd.to_datetime(start_date.dt.year.astype(str) + "-01-01"), #time since start of year
     data_27], #boolean representing whether event was epicfest
     axis=1) #join columns

#rename columns
data_compiled.columns =['start month', 'start day', 'duration', 'time since Jan 1', 'is Epicfest']

data_compiled.to_csv('training_data.csv')