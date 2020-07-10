#!/bin/bash

set -e

for j in `seq 2 24`; do
    rxna=$(($j * 36 + 100))
    rxnb=`expr "$rxna" + 9`
    rxnc=`expr "$rxnb" + 9`
    rxnd=`expr "$rxnc" + 9`
    cat node0.sh | sed -e "3s/0/$j/" -e "47s/100/$rxna/" -e "50s/109/$rxnb/" -e "53s/118/$rxnc/" -e "56s/127/$rxnd/" > node${j}.sh
    chmod +x node${j}.sh
done
