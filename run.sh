#!/bin/bash

date=`cat setting | grep date | cut -d: -f2`

echo "KMeans"

python3 20${date}_KMeans_MI_3.py
sleep 10s

python3 20${date}_KMeans_MI_5.py

sleep 10s

echo "GMM"
python3 20${date}_GMM_3.py

sleep 10s

python3 20${date}_GMM_5.py

sleep 10s

echo "Linkage_ALL"
python3 20${date}_Linkage_MI_3.py

sleep 10s

python3 20${date}_Linkage_MI_5.py

sleep 1s

echo "DONE"

exit 0
