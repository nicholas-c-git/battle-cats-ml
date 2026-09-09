import pandas as pd
import numpy as np

#used to turn 'series id' into a bunch of binary features
from sklearn.preprocessing import OneHotEncoder

#you can make series_id.csv using convert_series_id.py
#sorted by 'series id' then 'start date'
series_data = pd.read_csv('series_id.csv')

#date columns as a datetime
start_date = pd.to_datetime(series_data['start date']) #YYYY-MM-DD
end_date = pd.to_datetime(series_data['end date']) #YYYY-MM-DD

#make 'days since last appearance' column
    #each index's value = start date - previous index's start date
series_data['days since last appearance'] = start_date - start_date.shift(periods=1)

#for every item in series id, skipping index 0
for i in range(1,series_data.last_valid_index()):

    #get the series id for this index and the previous index
    series = series_data.at[i, 'series id']
    prev_series = series_data.at[i-1, 'series id']

    #remove the value if they are from different series
    if series != prev_series:
        series_data.at[i, 'days since last appearance'] = None

#uncomment to get the series_data as a csv
#series_data.to_csv('series_data.csv', index=True)

#making a dataframe of most of the important features
data_compiled = pd.concat(
    [start_date.dt.month,
     start_date.dt.day,
     end_date - start_date, #duration in days
     start_date - pd.to_datetime(start_date.dt.year.astype(str) + "-01-01"), #days since start of year
                                 #do not try to convert the above to using formatted strings, we are adding a string on purpose
     series_data['days since last appearance']
     ], axis=1) #join columns

#rename columns and convert number of seconds to number of days
data_compiled.columns = ['start month', 'start day', 'duration', 'days since Jan 1', 'days since last appearance']
data_compiled['duration'] = data_compiled['duration'].dt.days
data_compiled['days since Jan 1'] = data_compiled['days since Jan 1'].dt.days

#initialize a one hot encoder in pandas DataFrame format
one_hot = OneHotEncoder(sparse_output=False).set_output(transform='pandas')

#make a DataFrame of the encoded series ids columns
#convert_dtypes because the 1s and 0s were 1.0 and 0.0
data_encoded = one_hot.fit_transform(series_data[['series id']]).convert_dtypes()

data_compiled = pd.concat([data_compiled, data_encoded], axis=1)

#output the input features to a csv
#still sorted by 'series id' then 'start date'
data_compiled.to_csv('training_data.csv', index=False)