#!/bin/bash
set -e
source activate BayesRetro

reaction_num=0
mkdir -p results
cd results
find ../smc/results -maxdepth 1 -name "reaction${reaction_num}_*.pickle}" -exec ln -s {} \;
cd ..
python candidates_extracting.py $reaction_num
mkdir -p results_summary/cand_fps
python candidate_fingerprints_calculation.py $reaction_num

cd results_summary/cand_fps/
ln -s ../../utils/glmnet_grouped.RData
ln -s ../../utils/calc_prob.R
if [ -f cand_fps_rxn${reaction_num}.npz ]
then
    Rscript calc_prob.R $reaction_num
    echo "Done $reaction_num"
else
    echo "$reaction_num does not exit"
fi
cd ../..
python ranking.py $reaction_num
