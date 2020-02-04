set -e
echo $PATH
# Next line is an example for loading cuda environment. Comment out next line if cuda is already in your path.
module load cuda/9.2
conda create -y -n python36 python=3.6
source activate python36
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
