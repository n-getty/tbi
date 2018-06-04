import fnmatch
import os
import pandas as pd


fn = "/vol/ml/ngetty/fcon"
demos = []

for root, dirnames, filenames in os.walk(fn):
    for filename in fnmatch.filter(filenames, '*.txt'):
        file = os.path.join(root, filename)
        df = pd.read_csv(file, usecols=[0,2,3], names=['ID', 'Age', 'Sex'], delimiter='\t')
        if df.Age[0] in ['m', 'f']:
            df = pd.read_csv(file, usecols=[0, 3, 2], names=['ID', 'Age', 'Sex'], delimiter='\t')
        elif df.Age[0] == 'y':
            print file
            break
            
        demos.append(df)

demos = pd.concat(demos)

demos.to_csv('all_demos')