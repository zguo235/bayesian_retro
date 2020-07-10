#!/bin/bash

set -e

for j in `seq 1 9`; do
    # rxna=$(($j * 20))
    # rxnb=`expr "$rxna" + 5`
    # rxnc=`expr "$rxnb" + 5`
    # rxnd=`expr "$rxnc" + 5`
    cat node0.sh | sed -e "3s/0/$j/" -e "54s/ 0/ $j/" -e "57s/0/$j/" -e "60s/0/$j/" -e "63s/0/$j/" > node${j}.sh
    chmod +x node${j}.sh
done
