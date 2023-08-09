#!/bin/bash

set -e

if [ $# -lt 1 ]; then
    echo "Please pass the input csv file"
    exit 1
fi

FILE=$1

python3 ../analyze/cli.py -i $FILE \
  --power.init config 1 \
  -g Exp,hw_v,hw_g `#group per observable experiment use hw as otherwise we have more than 16k groups` \
  --per REnergyPP0,RTicks 5 95 `#remove outliers` \
  -u -g Exp `#regroup for CPA` \
  --power.set_comp all `#use all the possible model components in the linear regression` \
  --power.find_coefs RPowerPP0 0
