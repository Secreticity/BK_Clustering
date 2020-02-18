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

df = pd.read_csv("total_all_1G_pp_norm.csv")
df = df[df.columns[8:]]
df = df.drop('writeRateTotal', axis=1)

data_points = df.values

text_file = open("data/0204_ALL_KMeans_score.txt", "a+")
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
    
    text_file = open("data/0204_ALL_KMeans_score.txt", "a+")
    text_file.write("\n\nn_clusters ="+str(n_clusters)+"The average silhouette_score is :"+str(silhouette_avg))
    text_file.write("\nn_clusters ="+str(n_clusters)+"The average DBI_score is :"+str(DBI_avg))
    text_file.close()
    
    print("For n_clusters =", n_clusters,
      "The average silhouette_score is :", silhouette_avg)
    print("For n_clusters =", n_clusters,
      "The average DBI_score is :", DBI_avg)
    
    ### Save R for the cluster
    #r.to_csv("0204_ALL_KMeans_"+str(n_clusters)+".csv", mode='w')

# scatter plot
#fig = plt.figure( figsize=(10,10))
#ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

#'totalFile','totalOpenReq','totalReadReq'
#ax.scatter(r[flist[0]],r[flist[1]],r[flist[2]],c=r['predict'],alpha=0.5)
#ax.set_xlabel(flist[0])
#ax.set_ylabel(flist[1])
#ax.set_zlabel(flist[2])
#plt.show()
#plt.savefig("rand_plt/"+feature1+"_"+feature2+"_"+feature3+".png")
"DONE"

