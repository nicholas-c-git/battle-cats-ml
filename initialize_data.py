import pandas as pd
import numpy as np

from pathlib import Path

#testing read_table on a singular file, not used for anything else
data_test = pd.read_table(
                          'events/20180707.tsv', #one of the event/ files
                          on_bad_lines='skip', #skippping here, full data uses a function
                          header=None,
                          skiprows=1, #skipping the tsv [start] line
                          #skipfooter might be unnecessary when using dropna
                          engine='python', #engine is python for on_bad_lines handling
                          parse_dates=[0,2],#read the start and end dates as dates
                          date_format="%Y%m%d", #date format
                          #renaming columns
                          names=['start date','start hour','end date','end hour','min version','max version',
                                 'ignore1','ignore2','gacha type','num gachas','gacha id', 'gacha price','ignore3',
                                 'ignore4','chance normal','ignore5','chance rare','ignore6','chance super','ignore7',
                                 'chance uber','guaranteed event','chance legend','ignore8','gacha message'],
                          )

#this will be used in the next section
#used when a row is a bad line (has more that 25 columns)
#instead of skipping the bad line, this allows us to read just take the first 25 columns
#only works when read_table uses engine='python'
def read25(line:list(str)):
    return line[:25]

#full data
#turns the event/ files into a list of DataFrames, then combines all DataFrames in the list
data_full = pd.concat([
    pd.read_table(
                #keyword arguments copied from data_test
                file_names, on_bad_lines=read25, header=None, skiprows=1, engine='python',
                parse_dates=[0,2],date_format='%Y%m%d',
                names=['start date','start hour','end date','end hour','min version','max version',
                    'ignore1','ignore2','gacha type','num gachas','gacha id', 'gacha price','ignore3',
                    'ignore4','chance normal','ignore5','chance rare','ignore6','chance super','ignore7',
                    'chance uber','guaranteed event','chance legend','ignore8','gacha message'],
                )
    #do the read function for all files in the events/ directory
    for file_names in Path('events').iterdir()],

    #ignore index of the individual files when combining the list of DataFrames
    #now, the DataFrames are all combined
    #drop the empty footer rows
    #give the dataframe a fresh index
    #simplify data types
    ignore_index=True).dropna(subset=['gacha id']).reset_index().convert_dtypes()

#uncomment the next line for data_full.csv
#data_full.sort_values(by=['gacha id']).to_csv('data_full.csv', index=False)

#simplify the table to just what we will use
data_important = data_full[['start date', 'end date', 'gacha type', 'gacha id', 'gacha message']]

#add a new column, will be used for identifying banner, default value set as -1
data_important.insert(len(data_important.columns),'series id',-1)

#GatyaSetID is 'gacha id', seriesID is what we want and will become 'series id'
interpreter_table = pd.read_table('GatyaData_Option_SetR.tsv',index_col='GatyaSetID')

#iterate through data rows
for index in range(len(data_important.index)):
    #we use gacha_id to find the banner and gacha_type because we only want 'rare' gacha events
    gacha_id = data_important.at[index,'gacha id']
    gacha_type = data_important.at[index,'gacha type']

    #dropping indexes with gacha_id for empty or platinum capsules
    #dropping indexes with gacha_type of normal and event gachas
    if int(gacha_id) < 1 or gacha_type != 1:
        data_important.drop(index=index, inplace=True)

    #for indexes that we want to keep
    else:
        #fill in the series_id using the interpreter_table to match the 'gacha id' with the right 'series id'
        data_important.at[index,'series id'] = interpreter_table.at[gacha_id, 'seriesID']

#converting the 'start date' column values back to time format
#it unconverted at somepoint earlier, the reason is still unknown
data_important['start date'] = pd.to_datetime(data_important['start date'], format='%Y%m%d')

#output the table into a file
data_important.sort_values(by=['series id','start date']).to_csv('series.csv', index=False)



#additional things for later:
#a couple of valentines banners got marked as gacha id 475, which corresponds to ultra souls

#missing series ids: 11,12,17,21,25,29,36,45,46,55,58,59,60,62,69
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
