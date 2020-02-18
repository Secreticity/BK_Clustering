#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import sys

file_path = sys.argv[1]

if len(sys.argv) != 2:
    print("Insufficient arguments")
    sys.exit()

for iter in range(2,31):
    df = pd.read_csv(file_path+str(iter)+".csv")
    df2 = df["predict"].value_counts().to_frame(str(iter)+' clusters')#.sort_values(by=str(iter)+' clusters', ascending=False)
    
    if (iter == 2):
        r = df2
    else:
        r = pd.concat([r,df2],axis=1).fillna(0).astype(int)
        r[str(iter)+' clusters'] = r[str(iter)+' clusters'].sort_values(ascending=False).values

r.to_csv(file_path+"count_values.csv", mode='w')

