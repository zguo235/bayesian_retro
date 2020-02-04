# Bayesian Retrosynthesis

This is the code for the "Finding diverse routes of organic synthesis using surrogate-accelerated Bayesian retrosynthesis"

## Requirements

A new conda environment can be created by python36_environment.sh
```bash

./python36_environment.sh
```

## Sequential Monte Carlo algorithm for retrosynthesis

quick_start.sh is an example to try the SMC algorithm for retrosynthesis.
Each step of SMC takes about 30 sec. Step number is set to 600. The whole algorithm takes 6 h.
```bash

./quick_start.sh
```
## Ranking found reactions

The ranking model is in experiments_single-step/test0/postprocessing_ranking/ directory.
Use the results_analysis.sh to rank the found reactions. Final results will in reactants_rank_basedon_prob.csv.
```bash

./reactants_rank_basedon_prob.sh
```
