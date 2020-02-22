#!/bin/bash

echo "[ Result Extraction Script ]"
echo " "

if [ $# -ne 1 ]; then

	echo "./num_extract.sh [filename]"
	echo "Two Parameters input required!"
	exit 1
fi

echo "[silhoutte]"
cat ${1} | grep -E sil | cut -d: -f2

echo " "
echo "[DBI]"
cat ${1} | grep -E DBI | cut -d: -f2

exit 0
