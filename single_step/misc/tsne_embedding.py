import os
import sys
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from lvdmaaten import bhtsne
from utils.draw_utils import draw_mols_smi
from utils.ga_utils import csc_drop_zerocols


with open('data/candidates_single.txt') as f:
    candidates_smis = [s.rstrip() for s in f.readlines()]
n_candidates = len(candidates_smis)
candidates_smis = np.array(candidates_smis)
candidates_fps = sp.load_npz('data/candidates_fp_single.npz')
test_1step = pd.read_pickle('data/preprocessed_liu_dataset/test_sampled.pickle')

reaction_num = int(sys.argv[1])
result_num = 0

target_reaction = test_1step.iloc[reaction_num, :]
target_reactant_smi, target_product_smi = test_1step.iloc[reaction_num, [0, 1]]
target_reactant_smi = target_reactant_smi.split('.')
target_reactant_idx = list()
for smi_single in target_reactant_smi:
    idx_single = np.nonzero(candidates_smis == smi_single)[0][0]
    target_reactant_idx.append(idx_single)
target_reactant_idx = (tuple(sorted(target_reactant_idx)),)

summary_path = os.path.join('results_summary', 'reaction{}.pickle'.format(reaction_num))
with open(summary_path, 'rb') as f:
    summary_df = pickle.load(f)

try:
    cand_prob_path = os.path.join('results_summary', 'cand_fps', 'cand_prob_rxn{}.csv'.format(reaction_num))
    cand_prob = pd.read_csv(cand_prob_path, dtype=float)
    cand_prob = cand_prob.max(axis=1)
    summary_df_len = list(map(len, summary_df))
    summary_df_len = np.cumsum(summary_df_len)
    if len(cand_prob) == summary_df_len[-1]:
        cand_prob = np.split(cand_prob.values, summary_df_len[:-1])
    else:
        print("Length of candidate class prediction differs from total summary_df length.",
              file=sys.stderr, flush=True)
except FileNotFoundError:
    print('cand_prob_rxn{}.csv'.format(reaction_num), "doesn't exist")

df = summary_df[result_num]
print('Number of candidate synthetic routes:', len(df))
df['reactants_idx'] = df['reactants'].apply(lambda x: x.immutable_list)
df['prob'] = cand_prob[result_num]
df['prob_multi'] = np.exp(df['score'].values) * cand_prob[result_num]

df_fps = df['reactants'].apply(lambda x: x.idx2fp(candidates_fps))
df_fps = sp.csc_matrix(np.concatenate(df_fps.values, axis=0))
df_fps = csc_drop_zerocols(df_fps)
df_fps = np.asarray(df_fps.todense(), dtype=float)
fps_tsne = bhtsne.run_bh_tsne(df_fps, perplexity=50, theta=0.5,
                              initial_dims=df_fps.shape[1],
                              use_pca=True, verbose=1, max_iter=1000)
df['tsne_x'] = fps_tsne[:, 0]
df['tsne_y'] = fps_tsne[:, 1]

df_sorted = df.sort_values(by='prob_multi', axis=0, ascending=False).reset_index(drop=True)

save_dir = Path('tsne_embedding')
save_dir.mkdir(exist_ok=True)

df_smis = pd.DataFrame()
df_smis['reactant_smis'] = df_sorted['reactants'].apply(lambda x: x.idx2smi(candidates_smis)[0])
df_sorted['reactant_smis'] = df_smis['reactant_smis']
df_smis['product_smis'] = target_product_smi
df_sorted['product_smis'] = target_product_smi
df_sorted['true_reactant'] = df_sorted['reactants_idx'] == target_reactant_idx
df_pickle = df_sorted[['reactants_idx', 'reactant_smis', 'product_smis',
                       'distance_pred', 'distance_true', 'score', 'prob',
                       'prob_multi', 'tsne_x', 'tsne_y', 'true_reactant']]
df_pickle.to_pickle(str(save_dir / 'reaction{}_sorted.pickle'.format(reaction_num)))
