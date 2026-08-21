import pandas as pd
import numpy as np

from pathlib import Path

#successful test
data_test = pd.read_table(
                          'events/20180626.tsv', #first event file
                          on_bad_lines='skip', #skipping rows with more than 25 columns
                          header=None,
                          skiprows=1, #skipping the tsv [start] line
                          skipfooter=1, #skipping the tsv [end] line
                          #skipfooter might be unnecessary when using dropna
                          engine='python',
                          #dropping the rows with an empty banner name
                          ).dropna(subset=[24])
                          #0-index column 24 may or may not be banner name/title
print(data_test[24])

#trying to combine the event data files into one dataframe
#ValueError: Indexes have overlapping values...
data_full = pd.DataFrame().join([
    #engine is pyarrow because it has multithreading, pyarrow doesn't support skipfooter
    pd.read_table(file_names, on_bad_lines='skip', header=None, skiprows=1, engine='pyarrow').dropna(subset=[24])
    for file_names in Path('events').iterdir()
    ])

print(data_full)