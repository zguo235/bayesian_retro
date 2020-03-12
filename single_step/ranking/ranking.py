import os
import sys
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path


reaction_num = int(sys.argv[1])
with open('data/candidates_single.txt') as f:
    candidates_smis = [s.rstrip() for s in f.readlines()]
n_candidates = len(candidates_smis)
candidates_smis = np.array(candidates_smis)
candidates_fps = sp.load_npz('data/candidates_fp_single.npz')
test = pd.read_pickle('data/preprocessed_liu_dataset/test_sampled.pickle')
test = test.reset_index(drop=True)
summary_dir = Path('ranking_summary')
summary_dir.mkdir(exist_ok=True)

target_reactant_smi, target_product_smi = test.iloc[reaction_num, [0, 1]]
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
    cand_prob_path = os.path.join('results_summary', 'candidate_reactions_fps', 'reaction{}_prob.csv'.format(reaction_num))
    cand_prob = pd.read_csv(cand_prob_path, dtype=float)
    cand_prob = cand_prob.max(axis=1)
    summary_df_len = list(map(len, summary_df))
    summary_df_len = np.cumsum(summary_df_len)
    if len(cand_prob) == summary_df_len[-1]:
        cand_prob = np.split(cand_prob.values, summary_df_len[:-1])
    else:
        print("Length of candidate class prediction differs from total summary_df length.",
              file=sys.stderr, flush=True)
        sys.exit(1)
except FileNotFoundError:
    print('reaction{}_prob.csv'.format(reaction_num), "doesn't exist")
    sys.exit(0)

results_summary = list()
for i, df in enumerate(summary_df):
    if len(df) == 0:
        summary = [reaction_num, False, 0, False, None, None, None, None, None, None, None, None]
    else:
        df['reactants_idx'] = df['reactants'].apply(lambda x: x.immutable_list)
        df['prob'] = cand_prob[i]
        df['prob_multi'] = np.exp(df['score'].values) * cand_prob[i]
        if target_reactant_idx in set(df['reactants_idx']):
            df_sorted = df.sort_values(by='prob_multi', axis=0, ascending=False).reset_index(drop=True)
            true_reactant_order = df_sorted[df_sorted['reactants_idx'] == target_reactant_idx].index[0]
            retro_result_true = df[df['reactants_idx'] == target_reactant_idx].iloc[0, :]
            step_num = retro_result_true.name // 1000
            summary = [reaction_num, True, len(df), True, true_reactant_order, step_num]
            summary.extend(retro_result_true.loc[['identical_prod_id', 'score', 'distance_pred', 'distance_true', 'prob', 'prob_multi']])
        else:
            summary = [reaction_num, True, len(df), False, None, None, None, None, None, None, None, None]
    results_summary.append(summary)

summary = pd.DataFrame(results_summary, columns=['reaction_num', 'product_found', 'n_candidates', 'reactant_found',
                                                 'true_reactant_order', 'step_num', 'identical_prod_id', 'score',
                                                 'distance_pred', 'distance_true', 'prob_max', 'prob_multi'])
summary.to_csv(str(summary_dir / 'reaction{}.csv'.format(reaction_num)), index=False)
