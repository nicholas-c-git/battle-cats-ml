import pandas as pd
import numpy as np

from pathlib import Path

data_test = pd.read_table(
                          'events/20180626.tsv',
                          on_bad_lines='skip',
                          header=None,
                          skiprows=1, #skipping the tsv [start] line
                          skipfooter=1, #skipping the tsv [end] line
                          #skipfooter might be unnecessary with dropna
                          engine='python',
                          )

data_test.dropna(subset=[24],inplace=True)

print(data_test[24])

data_full = pd.DataFrame().join([
    #engine is pyarrow because it has multithreading, pyarrow doesn't support skipfooter
    pd.read_table(file_names, on_bad_lines='skip', header=None, skiprows=1, engine='pyarrow').dropna(subset=[24])
    for file_names in Path('events').iterdir()
    ])

print(data_full)