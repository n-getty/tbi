import fnmatch
import os
import pandas as pd


fn = "/vol/ml/ngetty/fcon"
demos = []

for root, dirnames, filenames in os.walk(fn):
    for filename in fnmatch.filter(filenames, '*.txt'):
        file = os.path.join(root, filename)
        df = pd.read_csv(file, usecols=[0,2,3], names=['ID', 'Age', 'Sex'])
        demos.append(df)

demos = pd.concat(demos)

demos.to_csv('all_demos')