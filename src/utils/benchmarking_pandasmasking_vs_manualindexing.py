"""
This is to compare query time of pandas masking vs. pivot trick vs. manual dictionary indexing.
pandas is much slower than manual dictionary, pivot trick is as fast as manually but confusing.
"""

import pandas as pd
import numpy as np
from copy import deepcopy
import time

ylength = 10000
clength = 100000
y = pd.DataFrame(np.zeros((ylength, 6)))
y.iloc[:, 0] = np.arange(ylength)
print(y.shape)
c = pd.DataFrame(np.zeros((clength, 3)))
c.iloc[:, 0] = np.random.randint(0, ylength, clength)
c.iloc[:, 1] = np.random.randint(1, 6, clength)
c.iloc[:, 2] = np.random.randint(1, 100, clength)
print(c.shape)
c = c.drop_duplicates(subset=c.columns[[0, 1]])  # Simulate real data, year and timestep are not duplicate.
print(c.shape)

def using_pandas(y,c):
    y = deepcopy(y)
    c = deepcopy(c)
    c = c.values
    for row in c:
        y.iloc[[y.iloc[:, 0] == row[0]], row[1]] = row[2]
    return y

def manually_indexing(y,c):
    y = deepcopy(y)
    c = deepcopy(c)
    y = y.values
    c = c.values
    # Build dict.
    data2rownum = {}
    for i, rowi in enumerate(y):
        data2rownum[rowi[0]] = i
    # Then, update.
    for row in c:
        if row[0] in data2rownum:
            y[data2rownum[row[0]], row[1]] = row[2]
    y = pd.DataFrame(y)
    return y

startp = time.time()
p = using_pandas(y,c)
stopp = time.time()

startm = time.time()
m = manually_indexing(y,c)
stopm = time.time()

# Using pivot table to convert c(pid, timestep, count) to c2(pid, count at timestep)
starti = time.time()
c = c.pivot_table(index=c.columns[[0]].tolist(), columns=c.columns[[1]].tolist(), values=c.columns[[2]].tolist(), aggfunc=np.mean)
c.columns = c.columns.droplevel(0)
c = c.reset_index(drop=False)
y = pd.DataFrame(np.arange(ylength))
y = pd.merge(y, c, left_on=y.columns[0], right_on=c.columns[0], how='left')
y = y.fillna(0.0)
stopi = time.time()

print('Check consistency pandas masking: ' + str((p.values==m.values).all().all()))
print('Check consistency pivot: ' + str((y.values==m.values).all().all()))
print('Using pandas time: ' + str(stopp-startp))
print('Pivot time: ' + str(stopi-starti))
print('Manually indexing time: ' + str(stopm-startm))

"""
Result:
y shape = (10000, 6)
c shape = (43000, 3)
Check consistency pandas masking: True
Check consistency pivot: True
Using pandas time: 77.9657349586
Pivot time: 0.0611219406128
Manually indexing time: 0.081906080246
"""
