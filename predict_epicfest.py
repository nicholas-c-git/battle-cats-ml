import pandas as pd
import numpy as np

#used for testing
import matplotlib.pyplot as plt

#needed for finding today's date, used when predicting
from datetime import date

#used for separating the testing set from the training set
from sklearn.model_selection import train_test_split

#distance based models like KNeighbors should have standardized input features
from sklearn.preprocessing import StandardScaler

#basic classification models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#you can make training_data.csv using prepare_training_data.py
prepared_data = pd.read_csv('training_data.csv')

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

#make the dictionary with reversed key-value pairs
name_to_series_id = {}
for id, name in series_id_to_name.items():
    name_to_series_id.setdefault(name, id)

#take a user input for the series id focus on
#   also some error/input handling

#title
print("\nNicholas's Battle Cats Banner Predictor!\n")

#series id 27 is epicfest
#works for other ids (only the ids in the dict below)
#also works for the exact names of the id
BANNER_ID = input("What banner do you want to look for?\n"+
                  "type 'help' for a list of valid inputs, default value is 27 (epicfest)\n")

#only if the user input is 'help'
while BANNER_ID == "help":

    #making a really long string with all of the series ids and their corresponding banner names
    key_value_pairs = ""

    #format the keys and items into the string
    for key,item in series_id_to_name.items():
        key_value_pairs += f"ID {key} is for '{item}'\n"

    #print the string and then ask for another input
    #   does this loop again until the user input is something other than 'help'
    print('\n' + key_value_pairs)
    BANNER_ID = input("Type either the banner name or its ID number\n")

#try to use the user input as the key for the series_id_to_name dict
#importantly, the dict takes an int, so we try to convert the user input to an int
try:
    BANNER_ID = int(BANNER_ID)
    print("\nseries id accepted\n")

#if the input wasn't an int, it will raise a ValueError
#we check if it is a valid banner name/a key to the name_to_series_id dict (flipped version of series_id_to_name)
except ValueError:
    if BANNER_ID.lower() in name_to_series_id.keys():
        BANNER_ID = name_to_series_id.get(BANNER_ID.lower())
        print("\nbanner id accepted\n")

    #if all of that fails, just use our default banner
    #our default is currently set to series id 27 (epicfest)
    else:
        BANNER_ID = 27
        print("invalid series id, defaulting to 27\n")

#a dataframe with 1 column named 'is Correct Banner'
#   the column's values are taken from prepared_data's column corresponding to our BANNER_ID
id_data = pd.DataFrame({'is Correct Banner': prepared_data[f"series id_{BANNER_ID}"]})

#take out all the series id columns from prepared_data, since we have id_data for the series id that we care about
num_series = len(series_id_to_name.keys()) #this assumes that series_id_to_name has keys for every series id
prepared_data = prepared_data.iloc[:,:-num_series] #all indexes, and the columns except the series columns, which are all at the end

#make a DataFrame with the training data columns
data = pd.concat([prepared_data, id_data], axis=1)

print(f"now let's set up a model for banner ID {str(BANNER_ID)}: {series_id_to_name.get(BANNER_ID)}\n")

#X and y are input and output features
#All_X is a list of all the available input features
#   The list excludes 'is Correct Banner'
All_X = data.columns.to_list()[0:-1]

#dict with keys: 0 to len(All_X)-1, values: All_X's values
All_X_dict = dict([(x, All_X[x]) for x in range(len(All_X))])

print("what input features to use?")
X_keys = input("type 'help' for a list of valid input features\n")

#BANNER_ID's help-loop, look there for more comments
#   Almost the same but replace BANNER_ID with X_keys, and series_id_to_name to All_X_dict
while X_keys == "help":

    key_value_pairs = ""
    for key,item in All_X_dict.items():
        key_value_pairs += f"{key} for '{item}'\n"

    #slightly different lines printed
    print('\n' + key_value_pairs, "\nPlease only type the numbers that you want to use")
    X_keys = input("(Example: '012' to use the first 3 input features)\n")


#X and y are the input and output features
#X can take the features: 'start month', 'start day', 'duration', 'days since Jan 1', 'days since last appearance'
#   but 'is Correct Banner' should be an output feature (y),
#   and 'duration' isn't really a good input feature
X = data[['start month', 'start day', 'duration', 'days since Jan 1', 'days since last appearance']]
y = np.ravel(data[['is Correct Banner']])

#making X and y (input and output features) for each model
X_LR = X[['start month', 'start day', 'days since Jan 1']]
y_LR = y

#make separate datasets, one for model training, one for model testing/scoring
X_train_LR, X_test_LR, y_train_LR, y_test_LR = train_test_split(X_LR, y_LR, test_size=1/6, random_state=568)

#model selection is done with user input

#user inputs an int that corresponds to a model
#   some models don't have model weights, like KNeighborsClassifier
model_type = input("what model?\n0:LogisticRegression, 1:KNeighborsClassifier\n")
model = None
model_weights = True

#try to turn the user input into an int
try:
    model_type = int(model_type)
except ValueError:
    model_type = None

#depending on what the user input was, we use different models
#   models that have parameters will also prompt the user for parameters
match model_type:
    case 0:
        model = LogisticRegression()
    case 1:
        model = KNeighborsClassifier(n_neighbors=input("n_neighbors="))
        model_weights = False
    case _:
        print("invalid input")
#not finished yet, the user input system works but isn't used yet

#initializing a model and fitting it to training data
LRModel = LogisticRegression()
LRModel.fit(X_train_LR,y_train_LR)

#scoring the model performance on the datasets
print()
print("Logistic Regression training score:" , round(LRModel.score(X_train_LR, y_train_LR), 4))
print("Logistic Regression testing score:" , round(LRModel.score(X_test_LR, y_test_LR), 4))
print("Logistic Regression full dataset score:" , round(LRModel.score(X_LR, y_LR), 4))
print()

#model weights
print("Logistic Regression model weights:")
print('[Intercept]' , X.columns.to_list(), "\n" , LRModel.intercept_ , LRModel.coef_)
print()

#important note about linear regression
#predict() predicts all 0s, possibly something wrong with how I implemented/fit the model
#   I'm guessing that the model predicts 0 for everything
#   maybe an error with reading training data?
#   maybe not enough data?

#notes from few days later:
#   after a couple days of thinking, I'm pretty sure that it's the bad implementation of input features for
#   this kind of model. Log Regression finds linear correlations, but the correlation here might not be linear

#KNeighbors model now
X_KN = X[['days since Jan 1']]
y_KN = y

X_train_KN, X_test_KN, y_train_KN, y_test_KN = train_test_split(X_KN, y_KN, test_size=1/6, random_state=568)

KNModel = KNeighborsClassifier(n_neighbors=5)
KNModel.fit(X_train_KN,y_train_KN)

print()
print("K-Neighbors training score:" , round(KNModel.score(X_train_KN, y_train_KN), 4))
print("K-Neighbors testing score:" , round(KNModel.score(X_test_KN, y_test_KN), 4))
print("K-Neighbors full dataset score:" , round(KNModel.score(X_KN, y_KN), 4))
print()

print("K-Neighbors doesn't use model weights")
print()

#figuring out matplotlib

delete_quotations_to_see_plot='''
#initializing a plot
fig = plt.figure()
plt.scatter(X_KN, y, color='black')

#the predictions of each day of the year
x_prob = pd.DataFrame({'days since Jan 1':range(400)})
x_prob['days since Jan 1'] = np.linspace(X_KN.min(), X_KN.max(), 400)
y_prob = KNModel.predict_proba(x_prob)[:,1]

plt.plot(x_prob, y_prob) #the predictions for each day of the year
plt.xlabel("days since Jan 1")
plt.ylabel("epicfest likelihood")
plt.show()
'''

#planning to implement more models, and also want to use KFold cross validation
#   will probably start after the section below

#the models are all above this comment
#predicting the next 60 days and taking the most likely start date(s) (in progress)

#the number of days into the future to examine
#   should also work if the range crosses into the next year
PREDICTION_RANGE = 60 #input("how many days to look ahead for prediction: ")

#make new a DataFrame that will contain the date (index) days from today
from_today = pd.DataFrame(columns=['dates'])

#for i in 0-59
for i in range(PREDICTION_RANGE):
    #set index i of from_today to i days from today
    from_today.at[ i , 'dates' ] = pd.to_datetime(date.today()) + pd.to_timedelta(f"{i} days")
#turn it into a datetime so the format doesn't also include seconds
from_today = pd.to_datetime(from_today['dates'])

#dataframe of the next 60 days' 'days since Jan 1' (may need to change depending on model's input features)
Future_X = pd.concat([
                        from_today - pd.to_datetime(from_today.dt.year.astype(str)+'-01-01'),
                    ], axis=1)

#rename columns using the model's input feature's column names
#the feature_names_in_ gives me a red line error warning, but running the script works fine
Future_X.columns = KNModel.feature_names_in_

#change 'days since Jan 1' from 'i days' into just the int
Future_X['days since Jan 1'] = Future_X['days since Jan 1'].dt.days

#add a column 'prediction' which contains the probability that the banner appears for each day
Future_X['prediction'] = KNModel.predict_proba(Future_X)[:,1]

print("likely appearances:")
#go through the predictions
for i, prediction in enumerate(Future_X['prediction']):
    #print each day with a probability over 0.5
    if Future_X.at[i, 'prediction'] > 0.5:
        print(f"{round(prediction,4)} probability of {series_id_to_name.get(BANNER_ID)} on {from_today.get(i)}")