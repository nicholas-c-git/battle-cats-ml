import pandas as pd
import numpy as np

from datetime import date

from sklearn.model_selection import train_test_split

#encoding series id to either 1 for the desired event, or 0
from sklearn.preprocessing import OneHotEncoder

#basic classification models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#preparing the data

#you can make series.csv using initialize_data.py
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
#works for other ids (only the ids in the dict below)
BANNER_ID = 27
id_data = data_encoded['series id_' + str(BANNER_ID)]

#dict for ID to standardized name conversion
series_id_to_name = {0:'nekolugas', 1:'dynamites', 2:'vajiras', 3:'galaxy gals', 4:'dragon emperors', 5:'red busters',
                     6:'ultra souls', 7:'dark heroes', 8:'halloween', 9:'christmas', 10:"year's end", 13:'merc storia',
                     14:'mola', 15:'shoumetsu toshi', 16:"new year's + LNY", 18:'almighties', 19:'uberfest',
                     20:'summer', 22:'air busters', 23:'madoka magica', 24:'iron legion', 26:'spring', 27:'epicfest',
                     28:'girls monsters', 30:'ultra selection + LNY', 31:'miracle selection + LNY', 32:'metal busters',
                     33:'elemental pixies', 34:'fate', 35:'best of the best', 37:'evangelion', 38:'bikkuriman',
                     39:'anniversary + LNY', 40:'street fighters', 41:'LNY + excellent selection', 42:'superfest',
                     43:'hatsune miku', 44:'evangelion angels', 47:'seasonal', 48:'valentine', 49:'ranma', 50:'legendfest',
                     51:'river city clash', 52:'white day', 53:'june bride', 54:'street fighter 2',
                     56:'colossus slayers', 57:'river city clash 2', 59:'busters', 61:'90M DL',
                     63:'rurouni kenshin', 64:'summer 2', 65:'summer 3', 66:'LNY + miracle/ultra selection',
                     67:'LNY + miracle selection', 68: 'LNY + miracle/ultra selection 2', 70:'koneko',
                     71:'special units?', 72:'baki', 73:'sonic', 74:'demon slayer'}

print("initializing for banner ID " + str(BANNER_ID) + ":" , series_id_to_name.get(BANNER_ID))

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

#data is pretty much fully prepared now
#moving onto the machine learning stuff

#X and y are the input and output features
#X can take the features: 'start month', 'start day', 'duration', 'days since Jan 1', 'is Correct Banner'
#   but 'is Correct Banner' should be an output feature (y),
#   and 'duration' isn't really a good input feature
X = data_compiled[['start month','start day', 'days since Jan 1']]
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
print()

#planning to implement more models, and also want to use KFold cross validation
#   will probably start after the section below

#models all done
#predicting the next 60 days and taking the most likely start date(s) (in progress)

#the number of days into the future to examine
#   should also work if the range crosses into the next year
PREDICTION_RANGE = 60

#make new a DataFrame that will contain the date (index) days from today
from_today = pd.DataFrame(columns=['dates'])

#for i in 0-59
for i in range(PREDICTION_RANGE):
    #set index i to i days from today
    from_today.at[ i , 'dates' ] = pd.to_datetime(date.today()) + pd.to_timedelta(f"{i} days")
from_today = pd.to_datetime(from_today['dates'])

#dataframe of the next 60 days' 'start month','start day', 'days since Jan 1'
Future_X = pd.concat([
                        from_today.dt.month,
                        from_today.dt.day,
                        from_today - pd.to_datetime(from_today.dt.year.astype(str)+'-01-01'),
                    ], axis=1)

#rename columns
Future_X.columns = ['start month','start day', 'days since Jan 1']

#change 'days since Jan 1' from 'i days' into just the int
Future_X['days since Jan 1'] = Future_X['days since Jan 1'].dt.days

#predictions = LRModel.predict(Future_X)

print(Future_X.head())
#print(predictions)
