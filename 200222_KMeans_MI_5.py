#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
import datetime
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
#%matplotlib inline

f = open('setting', 'r')
lines = f.readlines()
for line in lines:
    if line.find("date") != -1:
        datestamp = line.spit(":")[1].rstrip()
    if line.find("dataname") != -1:
        file_name = line.split(":")[1].srtrip()
f.close()

df = pd.read_csv(file_name)

flist = ["seqReadPct","totalOpenReq","totalFile","totalReadReq","totalIOReq","seqWritePct","totalMetaReq"]
#flist = ["totalReadReq","totalFile","totalFileSTDIO","stripeSize","writeLess1k","writeTimePOSIXonly","readLess1m"]
df = df[flist[0:5]]

data_points = df.values

text_file = open("data/"+datestamp+"_5_KMeans_score.txt", "a+")
text_file.write("\n************"+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"************\n")
text_file.close()

### Number of clusters
for n_clusters in range(2, 31):
    model = KMeans(n_clusters=n_clusters).fit(data_points)
    predict = pd.DataFrame(model.fit_predict(df))
    predict.columns=['predict']

    # concatenate labels to df as a new column
    r = pd.concat([df,predict],axis=1)
    #print(r.sample(1))
    r.to_csv("csv/"+datestamp+"_KMeans_5_"+str(n_clusters)+".csv")

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    
    # clusters
    silhouette_avg = silhouette_score(data_points, predict.values.ravel())
    DBI_avg = davies_bouldin_score(data_points, predict.values.ravel())
    
    text_file = open("data/"+datestamp+"_5_KMeans_score.txt", "a+")
    text_file.write("\n\nn_clusters ="+str(n_clusters)+"The average silhouette_score is :"+str(silhouette_avg))
    text_file.write("\nn_clusters ="+str(n_clusters)+"The average DBI_score is :"+str(DBI_avg))
    text_file.close()
    
    print("For n_clusters =", n_clusters,
      "The average silhouette_score is :", silhouette_avg)
    print("For n_clusters =", n_clusters,
      "The average DBI_score is :", DBI_avg)
