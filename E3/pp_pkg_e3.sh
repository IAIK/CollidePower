#!/bin/bash

set -e

if [ $# -lt 1 ]; then
    echo "Please pass the input csv file"
    exit 1
fi

FILE=$1
n="${2:-100}"

python3 ../analyze/cli.py -i $FILE \
  --power.init config 0 \
  -g Exp,v0,g0 `#group per observable experiment` \
  --per DPower 5 95 `#remove outliers` \
  -u -g Exp `#regroup for CPA` \
  --power.cpa DPower $n 3,5,10,20,30,50,100,200,300,500,750,1000,2000,3000,5000,10000,12000,13000,15000,20000,23000 `#perform CPA` \
  --idx bit_per_hour \
  --print --plot.line_xf '1st' 1 `#show results`


