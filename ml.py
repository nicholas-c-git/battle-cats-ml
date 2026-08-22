import pandas as pd
import numpy as np

from pathlib import Path

#testing
data_test = pd.read_table(
                          'events/20180626.tsv', #first event file
                          on_bad_lines='skip', #skipping rows with more than 25 columns
                          header=None,
                          skiprows=1, #skipping the tsv [start] line
                          skipfooter=1, #skipping the tsv [end] line
                          #skipfooter might be unnecessary when using dropna
                          engine='python',
                          parse_dates=[0,2],#read the start and end dates as dates
                          date_format="%Y%m%d", #date format
                          #renaming columns
                          names=['start date','start hour','end date','end hour','min version','max version',
                                 'ignore1','ignore2','gacha type','num gachas','gacha id', 'gacha price','ignore3',
                                 'ignore4','chance normal','ignore5','chance rare','ignore6','chance super','ignore7',
                                 'chance uber','guarunteed event','chance legend','ignore8','gacha message'],
                          )
                          
#dropping the rows with an empty gacha message
data_test.dropna(subset=['gacha message'], inplace=True)

print(data_test)


#full data (not ready)
'''
#trying to combine the event data files into one dataframe
#ValueError: Indexes have overlapping values...
data_full = pd.DataFrame().join([
    #engine is pyarrow because it has multithreading, pyarrow doesn't support skipfooter
    pd.read_table(file_names, on_bad_lines='skip', header=None, skiprows=1, engine='pyarrow').dropna(subset=[24])
    for file_names in Path('events').iterdir()
    ])

print(data_full)'''