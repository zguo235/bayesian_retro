import os
import sys
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from utils.fingerprint_utils import SparseFingerprintCsrMatrix

candidates_fps = sp.load_npz('data/candidates_fp_single.npz')
reaction_num = int(sys.argv[1])
test = pd.read_pickle('data/preprocessed_liu_dataset/test_sampled.pickle')
target_reactant_smi, target_product_smi = test.iloc[reaction_num, [0, 1]]

summary_path = os.path.join('results_summary/', 'reaction{}.pickle'.format(reaction_num))
with open(summary_path, 'rb') as f:
    summary_df = pickle.load(f)

cand_fps_all = list()
for df in summary_df:
    if len(df) > 0:
        # FIXME: Only for single step
        reactant_cand_fps = df['reactants'].apply(lambda x: x.idx2fp(candidates_fps))
        reactant_cand_fps = np.concatenate(reactant_cand_fps.values)
        reactant_cand_fps = sp.csr_matrix(reactant_cand_fps)
        prod_fps = SparseFingerprintCsrMatrix(smis=[target_product_smi] * len(df)).tocsr()
        cand_fps = sp.hstack([reactant_cand_fps, prod_fps], format='csc')
        cand_fps_all.append(cand_fps)

if len(cand_fps_all) > 0:
    cand_fps_all = sp.vstack(cand_fps_all, 'csc')
    sp.save_npz(os.path.join('results_summary', 'cand_fps', 'cand_fps_rxn{}'.format(reaction_num)), cand_fps_all)
