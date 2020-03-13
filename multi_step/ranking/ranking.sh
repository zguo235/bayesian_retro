#!/bin/bash
set -e
source activate BayesRetro

reaction_num=0
python candidates_extracting.py $reaction_num

cd results_summary/candidate_reactions_fps
ln -s ../../utils/glmnet_grouped.RData
ln -s ../../utils/calc_prob_multi.R calc_prob.R
if [ -f reaction${reaction_num}_step1.npz -a -f reaction${reaction_num}_step2.npz ]
then
    Rscript calc_prob.R $reaction_num
    echo "Reaction$reaction_num done"
else
    echo "Reaction$reaction_num fingerprint files didn't find"
fi
cd ../..
python ranking.py $reaction_num
