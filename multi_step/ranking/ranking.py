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
# import shutil
# from utils.draw_utils import draw_mols_smi


with open('data/candidates_single.txt') as f:
    candidates_smis = [s.rstrip() for s in f.readlines()]
n_candidates = len(candidates_smis)
candidates_smis = np.array(candidates_smis)
candidates_fps = sp.load_npz('data/candidates_fp_single.npz')
reactant_num_list = [2, 1]
test_2steps = pd.read_pickle('data/preprocessed_liu_dataset/test_2steps.pickle')

# reaction_num = 0
total_num = int(sys.argv[1])
summary_dir = Path('ranking_summary')
summary_dir.mkdir(exist_ok=True)
candidates_sorted_dir = summary_dir / 'candidates_sorted'
candidates_sorted_dir.mkdir(exist_ok=True)

target_reaction = test_2steps.iloc[reaction_num, :]
reactant_smi_list = list()
for i in range(len(reactant_num_list)):
    reactant_smi_list.append(target_reaction['reactant_step{}'.format(i+1)].split('.'))
product_smi_list = list()
for i in range(len(reactant_num_list)):
    product_smi_list.append(target_reaction['product_step{}'.format(i+1)].split('.'))
target_product_smi = product_smi_list[-1][0]
target_reactant_smi_list = list()
target_reactant_smi_list.append(reactant_smi_list[0])
for i in range(len(reactant_num_list) - 1):
    trs = set(reactant_smi_list[i+1]).difference(set(product_smi_list[i]))
    target_reactant_smi_list.append(list(trs))
target_reactant_idx = list()
for smi_list in target_reactant_smi_list:
    id_list = [np.nonzero(candidates_smis == smi)[0][0] for smi in smi_list]
    target_reactant_idx.append(id_list)
target_reactant_idx = tuple(tuple(sorted(reactant)) for reactant in target_reactant_idx)

summary_path = os.path.join('results_summary', 'reaction{}.pickle'.format(reaction_num))
with open(summary_path, 'rb') as f:
    summary_df, candidate_reactions_len = pickle.load(f)
try:
    cand_step1_prob_path = os.path.join('results_summary', 'candidate_reactions_fps', 'reaction{}_step1_prob.csv'.format(reaction_num))
    cand_step1_prob = pd.read_csv(cand_step1_prob_path, dtype=float)
    cand_step1_prob = cand_step1_prob.max(axis=1)
    cand_step2_prob_path = os.path.join('results_summary', 'candidate_reactions_fps', 'reaction{}_step2_prob.csv'.format(reaction_num))
    cand_step2_prob = pd.read_csv(cand_step2_prob_path, dtype=float)
    cand_step2_prob = cand_step2_prob.max(axis=1)
    if len(cand_step1_prob) == len(summary_df) and len(cand_step2_prob) == len(summary_df):
        pass
    else:
        print("Length of candidate class prediction differs from summary_df length.",
              file=sys.stderr, flush=True)
        sys.exit(1)
except FileNotFoundError:
    print('Probability prediction of reaction{} candidates'.format(reaction_num), "doesn't exist")
    sys.exit(0)

if len(summary_df) == 0:
    summary = [reaction_num, False, 0, False, None, None]
else:
    summary_df['reactants_idx'] = summary_df['reactants'].apply(lambda x: x.immutable_list)
    summary_df['prob_step1'] = cand_step1_prob
    summary_df['prob_step2'] = cand_step2_prob
    summary_df['prob_multi'] = np.exp(summary_df['score'].values) * cand_step1_prob * cand_step2_prob
    summary_df = summary_df.drop_duplicates(subset={'reactants_idx'}, keep='first', inplace=False)
    # t-SNE embedding based on molecular fingerprint
    summary_fps = summary_df['reactants'].apply(lambda x: x.idx2fp(candidates_fps))
    summary_fps = sp.csc_matrix(np.concatenate(summary_fps.values, axis=0))
    summary_fps_dropped = csc_drop_zerocols(summary_fps)
    summary_fps_tsne = np.asarray(summary_fps_dropped.todense(), dtype=float)
    fps_tsne = bhtsne.run_bh_tsne(summary_fps_tsne, perplexity=50, theta=0.5,
                                  initial_dims=summary_fps_dropped.shape[1],
                                  use_pca=True, verbose=1, max_iter=1000)
    summary_df['tsne_x'] = fps_tsne[:, 0]
    summary_df['tsne_y'] = fps_tsne[:, 1]

    df_sorted = summary_df.sort_values(by='prob_multi', axis=0, ascending=False).reset_index(drop=True)
    df_pickle = df_sorted[['reactants_idx', 'reactant_smi_step1', 'product_smi_step1',
                        'reactant_smi_step2', 'product_smi_step2', 'distance_pred',
                        'distance_true', 'score', 'prob_step1', 'prob_step2',
                        'prob_multi', 'tsne_x', 'tsne_y']]
    df_pickle.to_pickle(str(candidates_sorted_dir / 'reaction{}.pickle'.format(reaction_num)))

    df_step1_smis = df_pickle['reactant_smi_step1'].str.split('.', expand=True)
    df_step1_smis.columns = ['reactant1_step1', 'reactant2_step1']
    df_step1_smis['product_step1'] = df_pickle['product_smi_step1']
    df_step2_smis = df_pickle['reactant_smi_step2'].str.split('.', expand=True)
    df_step2_smis.columns = ['reactant1_step2', 'reactant2_step2']
    df_step2_smis['product_step2'] = df_pickle['product_smi_step2']
    df_smis = df_step1_smis.join(df_step2_smis)
    df_smis = df_smis[['reactant1_step1', 'reactant2_step1', 'reactant1_step2', 'reactant2_step2', 'product_step2']]
    df_smis.iloc[:10].to_csv(str(summary_dir / 'reaction{}_top10.csv'.format(reaction_num)))
    smi_list_top5 = df_smis.iloc[:5].values.flatten()
    cand_top5_svg = draw_mols_smi(smi_list_top5, molsPerRow=5, subImgSize=(300, 300))
    smi_list_top10 = df_smis.iloc[5:10].values.flatten()
    cand_top10_svg = draw_mols_smi(smi_list_top10, molsPerRow=5, subImgSize=(300, 300))
    with open(str(summary_dir / 'reaction{}_top5.svg'.format(reaction_num)), 'wt') as f:
        f.write(cand_top5_svg)
    with open(str(summary_dir / 'reaction{}_top6-10.svg'.format(reaction_num)), 'wt') as f:
        f.write(cand_top10_svg)
    if target_reactant_idx in set(df_sorted['reactants_idx']):
        true_reactant = df_sorted[df_sorted['reactants_idx'] == target_reactant_idx].iloc[0]
        summary = [reaction_num, True, len(summary_df), True, true_reactant.name + 1, true_reactant.prob_multi]
    else:
        summary = [reaction_num, True, len(summary_df), False, None, None]


summary = pd.DataFrame([summary], columns=['reaction_num', 'product_found', 'n_candidates', 'reactant_found',
                                                 'true_reactant_order',  'prob_multi'])
summary = summary.set_index('reaction_num')
summary = target_reaction.join(summary)
summary.to_csv(str(summary_dir / 'reaction{}.csv'.format(reaction_num)), index=False)
