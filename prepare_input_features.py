import pandas as pd
import numpy as np

#used to turn 'series id' into a bunch of binary features
from sklearn.preprocessing import OneHotEncoder

#you can make series.csv using convert_series_id.py
data_important = pd.read_csv('series_id.csv').sort_values('start date')

#date columns as a datetime
start_date = pd.to_datetime(data_important['start date'])
end_date = pd.to_datetime(data_important['end date'])

#initialize a one hot encoder in pandas DataFrame format
one_hot = OneHotEncoder(sparse_output=False).set_output(transform='pandas')

#make a DataFrame of the series id column but encoded
#convert_dtypes because the 1s and 0s were 1.0 and 0.0
data_encoded = one_hot.fit_transform(data_important[['series id']]).convert_dtypes()


#making a dataframe of the features to be used
data_compiled = pd.concat(
    [start_date.dt.month,
     start_date.dt.day,
     end_date - start_date, #duration in days
     start_date - pd.to_datetime(start_date.dt.year.astype(str) + "-01-01"), #days since start of year
     id_data], #boolean representing whether event was the right event
     axis=1) #join columns

#rename columns and convert seconds to days
data_compiled.columns = ['start month', 'start day', 'duration', 'days since Jan 1', 'is Correct Banner']
data_compiled['duration'] = data_compiled['duration'].dt.days
data_compiled['days since Jan 1'] = data_compiled['days since Jan 1'].dt.days

#uncomment for csv
#data_compiled.to_csv('training_data.csv')