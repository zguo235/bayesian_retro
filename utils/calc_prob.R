# install.packages("glmnet")
# install.packages("reticulate")
rm(list=ls())
suppressMessages(library(glmnet))
suppressMessages(library(reticulate))
load('glmnet_grouped.RData')

use_condaenv('BayesRetro')
# py_config()
np <- import('numpy', convert=T)
sp <- import('scipy.sparse', convert=T)

args <- commandArgs(trailingOnly=TRUE)
reaction_num <- as.integer(args[1])
# reaction_num <- 36
cand_fps_path <- paste0('cand_fps_rxn', reaction_num, '.npz')
cand_fps <- sp$load_npz(cand_fps_path)

cand_prob  <- predict(fit_grouped, newx=cand_fps, type="response", s=best_lambda)
output_path  <- paste0('cand_prob_rxn', reaction_num, '.csv')
write.csv(cand_prob, file=output_path, row.names=F)
