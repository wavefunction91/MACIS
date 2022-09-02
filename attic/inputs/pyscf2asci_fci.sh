#!/bin/bash

nlines=`wc -l $1 | awk '{print $1}'`
tail -n $(expr $nlines - 0) $1 > tmp.xxx
awk '{print $2" "$3" "$4" "$5" "$1}' tmp.xxx > $2
rm tmp.xxx
