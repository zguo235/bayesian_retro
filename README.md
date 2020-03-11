# Bayesian Retrosynthesis

This is the code for the "A Bayesian retrosynthesis algorithm"

## Requirements

The forward prediction model (fine-tuned Molecular Transformer), ranking model (glmnet) and files storing nearest neighbors of each reactant candidate can be found [here](https://figshare.com/projects/bayesian_retro/76935)
Download these files and put them to appropriate directories.
```bash

wget -O nearest_neighbor.zip https://ndownloader.figshare.com/articles/11954913/versions/1
unzip nearest_neighbor.zip -d data/
rm nearest_neighbor.zip
mkdir -p forward_models
wget -O forward_models/fine_tuned_model_on_liu_dataset_0p02.pt https://ndownloader.figshare.com/files/21945630
wget -O utils/glmnet_grouped.RData https://ndownloader.figshare.com/files/21947469
```

A new conda environment can be created by BayesRetro_environment.sh
```bash

./BayesRetro_environment.sh
```

## Sequential Monte Carlo algorithm for retrosynthesis

quick_start.sh is an example to try the SMC algorithm for retrosynthesis.
Each step of SMC takes about 30 sec. Step number is set to 600. The total search will take 6 h .
```bash

./quick_start.sh
```
## Ranking candidate synthetic routes

The ranking model is in single-step/ranking directory.
Install R packages **glmnet** and **reticulate** from CRAN to use this model.
```{r}
install.packages("glmnet")
install.packages("reticulate")
```

Rank the detected synthetic routes
```bash

./ranking.sh
```
