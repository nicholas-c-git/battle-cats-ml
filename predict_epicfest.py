import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

#encoding series id to either 1 for the desired event, or 0
from sklearn.preprocessing import OneHotEncoder

#basic classification models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data_important = pd.read_csv('series.csv').sort_values('start date')

#date columns as a datetime
start_date = pd.to_datetime(data_important['start date'])
end_date = pd.to_datetime(data_important['end date'])

#initialize a one hot encoder in pandas DataFrame format
one_hot = OneHotEncoder(sparse_output=False).set_output(transform='pandas')

#make a DataFrame of the series id column but encoded
#convert_dtypes because the 1s and 0s were 1.0 and 0.0
data_encoded = one_hot.fit_transform(data_important[['series id']]).convert_dtypes()

#series id 27 is epicfest
banner_id = 27
id_data = data_encoded['series id_' + str(banner_id)]

#making a dataframe of the features to be used
data_compiled = pd.concat(
    [start_date.dt.month,
     start_date.dt.day,
     end_date - start_date, #duration in days
     start_date - pd.to_datetime(start_date.dt.year.astype(str) + "-01-01"), #days since start of year
     id_data], #boolean representing whether event was the right event
     axis=1) #join columns

#rename columns
data_compiled.columns = ['start month', 'start day', 'duration', 'days since Jan 1', 'is Correct Banner']
data_compiled['duration'] = data_compiled['duration'].dt.days
data_compiled['days since Jan 1'] = data_compiled['days since Jan 1'].dt.days

#data_compiled.convert_dtypes()

#uncomment for csv
data_compiled.to_csv('training_data.csv')

#X and y are the input and output features
X = data_compiled[['start month','start day', 'duration', 'days since Jan 1']]
y = np.ravel(data_compiled[['is Correct Banner']])

#make separate datasets, one for model training, one for model testing/scoring
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/6, random_state=568)

#initializing a model and fitting it to training data
LRModel = LogisticRegression()
LRModel.fit(X_train,y_train)

#scoring the model performance on the datasets
print()
print("Logistic Regression training score:" , round(LRModel.score(X_train, y_train), 4))
print("Logistic Regression testing score:" , round(LRModel.score(X_test, y_test), 4))
print("Logistic Regression full dataset score:" , round(LRModel.score(X, y), 4))
print()

#model weights
print("Logistic Regression model weights:")
print('[Intercept]' , X.columns.to_list(), "\n" , LRModel.intercept_ , LRModel.coef_)