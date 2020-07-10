#!/bin/bash
# set -e
source /etc/profile.d/modules.sh
module load cuda/9.2
export PATH="$HOME/miniconda/bin:$PATH"
source $HOME/miniconda/bin/activate
conda activate python36

# for reaction_num in `seq 0 9`;
# do
#     python candidates_extracting.py $reaction_num
# done

python candidates_extracting.py 1

# cd results_summary/candidate_reactions_fps
# ln -s ../../utils/glmnet_grouped.RData
# ln -s ../../utils/calc_prob_multi.R calc_prob.R
# if [ -f reaction${reaction_num}_step1.npz -a -f reaction${reaction_num}_step2.npz ]
# then
#     Rscript calc_prob.R $reaction_num
#     echo "Reaction$reaction_num done"
# else
#     echo "Reaction$reaction_num fingerprint files didn't find"
# fi
# cd ../..
# python ranking.py $reaction_num
