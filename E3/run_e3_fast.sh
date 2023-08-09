#!/bin/bash
set -e

if [ $# -lt 1 ]; then
    echo "Please pass the output csv file"
    exit 1
fi

FILE=$1

make -C ../poc/module clean all load
EXP=-DPAPER_FIGURE_9_FAST  make -C ../poc/user/ clean all
sudo ../poc/user/main 2> $FILE
head $FILE