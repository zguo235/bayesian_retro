#!/bin/bash
set -e
# source activate BayesRetro
export PATH="$HOME/miniconda/bin:$PATH"
source $HOME/miniconda/bin/activate
conda activate python36

# Run one time for the first reaction
# mkdir -p results
# cd results
# find ../../test* -name 'reaction*.pickle' -exec ln -s {} . \;
# cd ..

reaction_num=$0
python candidates_extracting.py $reaction_num

cd results_summary/candidate_reactions_fps
ln -s ../../utils/glmnet_grouped.RData
ln -s ../../utils/calc_prob_single.R calc_prob.R
if [ -f reaction${reaction_num}.npz ]
then
    Rscript calc_prob.R $reaction_num
    echo "Reaction$reaction_num done"
else
    echo "Reaction$reaction_num fingerprint files didn't find"
fi
cd ../..
python ranking.py $reaction_num
