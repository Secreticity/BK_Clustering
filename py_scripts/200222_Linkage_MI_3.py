#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score
import datetime

f = open('setting', 'r')
lines = f.readlines()
for line in lines:
    if line.find("date") != -1:
        datestamp = line.split(":")[1].rstrip()
    if line.find("flist") != -1:
        flist = list(line.split(":")[1].rstrip().split(" "))
    if line.find("dataname") != -1:
        file_name = line.split(":")[1].rstrip()
f.close()

df = pd.read_csv(file_name)

#flist = ["seqReadPct","totalOpenReq","totalFile","totalReadReq","totalIOReq","seqWritePct","totalMetaReq"]
#flist = ["totalReadReq","totalFile","totalFileSTDIO","stripeSize","writeLess1k","writeTimePOSIXonly","readLess1m"]
df = df[flist[0:3]]

mode = ['complete','average','ward']

for mode in mode:
    
    text_file = open("data/"+datestamp+"_3_"+mode+"Linkage_score.txt", "a+")
    text_file.write("\n************"+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"************\n")
    print("\n************"+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"************\n")
    text_file.close()
    
    ### Number of clusters
#    ranged = [2, 3, 4, 12]

    for n_clusters in range(2,31):

        ## linkage{"ward","complete","average","single"}, optional (default="ward")
        model = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage=mode)
        predict = pd.DataFrame(model.fit_predict(df))
        predict.columns=['predict']

        # concatenate labels to df as a new column
        r = pd.concat([df,predict],axis=1)

        r.to_csv("csv/"+datestamp+"_Linkage_"+mode+"_3_"+str(n_clusters)+".csv")

        #print(r.sample(10))
        # clusters
        silhouette_avg = silhouette_score(df.values, predict.values.ravel())
        DBI_avg = davies_bouldin_score(df.values, predict.values.ravel())
        text_file = open("data/"+datestamp+"_3_"+mode+"Linkage_score.txt", "a+")
        text_file.write("\n\nn_clusters ="+str(n_clusters)+"The average silhouette_score is :"+str(silhouette_avg))
        text_file.write("\nn_clusters ="+str(n_clusters)+"The average DBI_score is :"+str(DBI_avg))
        text_file.close()
        
        print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
        print("For n_clusters =", n_clusters, "The average DBI_score is :", DBI_avg)
