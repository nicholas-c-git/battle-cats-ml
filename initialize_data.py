import pandas as pd
import numpy as np

from pathlib import Path

#testing read_table on a singular file, not used for anything else
data_test = pd.read_table(
                          'events/20180707.tsv', #one of the event/ files
                          on_bad_lines='skip', #skipping rows with more than 25 columns
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
                          
#full data (not fully ready)
#trying to combine the event data files into one dataframe
#only works for data before mid 2021, tsv format might've changed after that
data_full = pd.concat([
    pd.read_table(
                #keyword arguments copied from data_test
                file_names, on_bad_lines='skip', header=None, skiprows=1, engine='pyarrow',
                parse_dates=[0,2],date_format='%Y%m%d',
                names=['start date','start hour','end date','end hour','min version','max version',
                    'ignore1','ignore2','gacha type','num gachas','gacha id', 'gacha price','ignore3',
                    'ignore4','chance normal','ignore5','chance rare','ignore6','chance super','ignore7',
                    'chance uber','guaranteed event','chance legend','ignore8','gacha message'],
                )
    #do the read function for all files in the events/ directory
    for file_names in Path('events').iterdir()
    #ignore index of the individual files
    #once combined, drop duplicates (some events have duplicates across the tsv files) and give the dataframe a fresh index
    ], ignore_index=True).drop_duplicates().reset_index()

#simplify the table to just what we will use
data_important = data_full[['start date', 'end date', 'gacha id', 'gacha message']]

#add a new column, will be used for identifying banner
data_important.insert(len(data_important.columns),'series id',-1)

#GatyaSetID is 'gacha id', seriesID is what we want and will become 'series id'
interpreter_table = pd.read_table('GatyaData_Option_SetR.tsv',index_col='GatyaSetID')

#for i in the data's rows
for i in range(len(data_important.index)):
    #fill in the column we added, using the interpreter_table to match the 'gacha id' with its 'series id'
    data_important.at[i,'series id'] = interpreter_table.at[data_important.at[i,'gacha id'], 'seriesID']

#output the table into a file
data_important.sort_values(by='series id').to_csv('full.csv', index=False)
