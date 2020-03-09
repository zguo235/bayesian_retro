set -e

# Load cuda and conda path.
# source /etc/profile.d/modules.sh
# module load cuda/9.2
# export PATH="$HOME/miniconda3/bin:$PATH"
# source $HOME/miniconda3/bin/activate

# Create a new conda environment.
conda create -y -n BayesRetro python=3.6
source activate BayesRetro
conda install -y rdkit -c rdkit
conda install -y future tqdm
conda install -y pytorch=0.4.1 torchvision cudatoolkit=9.0 -c pytorch
git clone https://github.com/pschwllr/MolecularTransformer.git
cd MolecularTransformer
pip install torchtext
pip install .
conda install -y scipy scikit-learn
conda install -y matplotlib seaborn
pip install lightgbm # --install-option=--gpu
cd ..
rm -rf MolecularTransformer
