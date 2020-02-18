#!/bin/bash

echo "[ Result Extraction Script ]"
echo " "
echo " "

if [ $# -ne 2 ]; then

	echo "./num_extract.sh [filename] [sil/DBI]"
	echo "Two Parameters input required!"
	exit 1
fi

cat ${1} | grep -E ${2} | cut -d: -f2

exit 0
