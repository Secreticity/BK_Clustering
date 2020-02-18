#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
import datetime
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
#%matplotlib inline

df = pd.read_csv("total_all_1G_pp_PCA.csv")

df = df[df.columns[7:]]
df = df.drop('writeRateTotal', axis=1)

data_points = df.values

text_file = open("data/0204_PCA_KMeans_score.txt", "a+")
text_file.write("\n************"+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"************\n")
text_file.close()

### Number of clusters
for n_clusters in range(31, 61):
    model = KMeans(n_clusters=n_clusters).fit(data_points)
    predict = pd.DataFrame(model.fit_predict(df))
    predict.columns=['predict']

    # concatenate labels to df as a new column
    r = pd.concat([df,predict],axis=1)
    #print(r.sample(1))

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    
    # clusters
    silhouette_avg = silhouette_score(data_points, predict.values.ravel())
    DBI_avg = davies_bouldin_score(data_points, predict.values.ravel())
    
    text_file = open("data/0204_PCA_KMeans_score.txt", "a+")
    text_file.write("\n\nn_clusters ="+str(n_clusters)+"The average silhouette_score is :"+str(silhouette_avg))
    text_file.write("\nn_clusters ="+str(n_clusters)+"The average DBI_score is :"+str(DBI_avg))
    text_file.close()
    
    print("For n_clusters =", n_clusters,
      "The average silhouette_score is :", silhouette_avg)
    print("For n_clusters =", n_clusters,
      "The average DBI_score is :", DBI_avg)
    
    ### Save R for the cluster
    #r.to_csv("data/0204_PCA_KMeans_"+str(n_clusters)+".csv", mode='w')

"DONE"
