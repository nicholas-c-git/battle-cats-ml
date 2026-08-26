import pandas as pd
import numpy as np

from pathlib import Path

#testing
data_test = pd.read_table(
                          'events/20180707.tsv', #one of the event/ files
                          on_bad_lines='warn', #skipping rows with more than 25 columns #changed to warn
                          header=None,
                          skiprows=1, #skipping the tsv [start] line
                          #skipfooter might be unnecessary when using dropna
                          engine='pyarrow', #engine is pyarrow because it has multithreading
                          parse_dates=[0,2],#read the start and end dates as dates
                          date_format="%Y%m%d", #date format
                          #renaming columns
                          names=['start date','start hour','end date','end hour','min version','max version',
                                 'ignore1','ignore2','gacha type','num gachas','gacha id', 'gacha price','ignore3',
                                 'ignore4','chance normal','ignore5','chance rare','ignore6','chance super','ignore7',
                                 'chance uber','guaranteed event','chance legend','ignore8','gacha message'],
                          )
                          
#dropping the rows with an empty gacha message
data_test.dropna(subset=['gacha message'], inplace=True)

#dropping the empty columns
data_test.drop(columns=['ignore1','ignore2','ignore3','ignore4','ignore5','ignore6','ignore7','ignore8'], inplace=True)

#dropping columns that won't be used
data_test.drop(columns=['start hour','end hour','min version','max version',
                        'num gachas','gacha price',
                        'chance normal','chance rare','chance super',
                        'chance uber','guaranteed event','chance legend'], inplace=True)

print(data_test)

#full data (not ready)
#trying to combine the event data files into one dataframe
#only works for data before mid 2021, tsv format might've changed after that

data_full = pd.concat([
    pd.read_table(
                #keyword arguments copied from data_test
                file_names, on_bad_lines='warn', header=None, skiprows=1, engine='pyarrow',
                parse_dates=[0,2],date_format='%Y%m%d',
                names=['start date','start hour','end date','end hour','min version','max version',
                    'ignore1','ignore2','gacha type','num gachas','gacha id', 'gacha price','ignore3',
                    'ignore4','chance normal','ignore5','chance rare','ignore6','chance super','ignore7',
                    'chance uber','guaranteed event','chance legend','ignore8','gacha message'],
                ).dropna(subset=['gacha message']
                ).drop(columns=['ignore1','ignore2','ignore3','ignore4','ignore5','ignore6','ignore7','ignore8']
                ).drop(columns=['start hour','end hour','min version','max version',
                        'num gachas','gacha price',
                        'chance normal','chance rare','chance super',
                        'chance uber','guaranteed event','chance legend'])
    for file_names in Path('events').iterdir()
    ], ignore_index=True).drop_duplicates()

print(data_full)

data_full.sort_values(by='start date').to_csv('full.csv', index=False)