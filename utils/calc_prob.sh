#!/bin/bash

# set -e
# source activate python36
set +e

for j in $(seq 0 $1); do
    fps_file=cand_fps_rxn$j.npz
    if [ -f $fps_file ]
    then
        Rscript calc_prob.R $j
        echo "Done $j"
    else
        echo "$fps_file don't exist"
    fi
done

set -e
