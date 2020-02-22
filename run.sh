#!/bin/bash

echo "200204_KMeans_ALL"
python3 200221_KMeans_MI_3.py

sleep 10s

python3 200221_KMeans_MI_5.py

sleep 10s

echo "200204_GMM_PCA"
python3 200221_GMM_3.py

sleep 10s

python3 200221_GMM_5.py

sleep 10s

echo "200204_Linkage_ALL"
python3 200221_Linkage_MI_3.py

sleep 10s

python3 200221_Linkage_MI_5.py

sleep 1s

echo "DONE"

exit 0
