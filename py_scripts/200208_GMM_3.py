#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score
import datetime

df = pd.read_csv("total_all_1G_pp_norm.csv")

flist = ["seqReadPct","totalOpenReq","totalFile","totalReadReq","totalIOReq","seqWritePct","totalMetaReq"]
#flist = ["totalReadReq","totalFile","totalFileSTDIO","stripeSize","writeLess1k","writeTimePOSIXonly","readLess1m"]
df = df[flist[0:3]]

text_file = open("data/0208_3_GMM_score.txt", "a+")
text_file.write("\n************"+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+"************\n")
text_file.close()

### Number of clusters
for n_clusters in range(2,31):

    db = GaussianMixture(n_components=n_clusters, covariance_type='full', max_iter=100, n_init=1)
    predict = pd.DataFrame(db.fit_predict(df))
    predict.columns=['predict']

    # concatenate labels to df as a new column
    r = pd.concat([df,predict],axis=1)

    #print(r.sample(10))
    # clusters
    silhouette_avg = silhouette_score(df.values, predict.values.ravel())
    DBI_avg = davies_bouldin_score(df.values, predict.values.ravel())
    
    text_file = open("data/0208_3_GMM_score.txt", "a+")
    text_file.write("\n\nn_clusters ="+str(n_clusters)+"The average silhouette_score is :"+str(silhouette_avg))
    text_file.write("\nn_clusters ="+str(n_clusters)+"The average DBI_score is :"+str(DBI_avg))
    text_file.close()
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
    print("For n_clusters =", n_clusters, "The average DBI_score is :", DBI_avg)

