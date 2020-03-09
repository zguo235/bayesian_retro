#!/bin/bash
set -e
source activate python36

python results_analysis.py 0
mkdir -p results_summary/cand_fps
python calc_cand_fps.py 0

cd results_summary/cand_fps/
ln -s ../../utils/glmnet_grouped.RData
ln -s ../../utils/calc_prob.R
ln -s ../../utils/calc_prob.sh
source calc_prob.sh 1
cd ../..
python ranking_basedon_prob.py 1
