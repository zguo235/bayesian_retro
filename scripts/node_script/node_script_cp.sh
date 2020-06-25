#!/bin/bash

set -e

for j in `seq 1 5`; do
    rxna=$(($j * 20))
    rxnb=`expr "$rxna" + 5`
    rxnc=`expr "$rxnb" + 5`
    rxnd=`expr "$rxnc" + 5`
    cat node0.sh | sed -e "3s/0/$j/" -e "54s/ 0/ $rxna/" -e "57s/5/$rxnb/" -e "60s/10/$rxnc/" -e "63s/15/$rxnd/" > node${j}.sh
    chmod +x node${j}.sh
done
