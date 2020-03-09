import os
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import shutil
from utils.draw_utils import draw_mol_smi
from utils.draw_utils import draw_mols_smi

reaction_num = int(sys.argv[1])
# # reaction_num = 60
test = pd.read_pickle('data/preprocessed_liu_dataset/test_sampled.pickle')
target_reactant_smi, target_product_smi = test.iloc[reaction_num, [0, 1]]
os.makedirs('results_figs', exist_ok=True)

#----------------------------------------------------------------------------------------------------
# Draw reactant and product
target_reactant_svg = draw_mol_smi(target_reactant_smi, legend='reaction{} reactant'.format(reaction_num))
target_product_svg = draw_mol_smi(target_product_smi, legend='reaction{} product'.format(reaction_num))
fig_path = Path('results_figs')
fig_reactant = fig_path / 'reaction{}_reactant.svg'.format(reaction_num)
fig_product = fig_path / 'reaction{}_product.svg'.format(reaction_num)
with fig_reactant.open('wt') as f:
    f.write(target_reactant_svg)
with fig_product.open('wt') as f:
    f.write(target_product_svg)
# Surrogate model accuracy
# fig_path = Path('./experiments/test{}/figures'.format(reaction_num // 8))
# for folder in fig_path.iterdir():
#     if folder.name.startswith('reaction{}_'.format(reaction_num)):
#         shutil.copy(str(folder / 'LightGBM_test.png'),
#                     'results_figs/reaction{}_lgb.png'.format(reaction_num))
#         break
#----------------------------------------------------------------------------------------------------


def result_analyzer(reaction_num, result, target_reactant_idx, target_product_smi):
    reactants_df, products, scores = zip(*result)
    products = [np.array(p) for p in products]
    scores = [np.array(s) for s in scores]
    reactants_df = pd.concat(reactants_df, ignore_index=True)
    products = np.concatenate(products)
    scores = np.concatenate(scores)

    reactants_df['prod_pred'] = products[:, 0]
    reactants_df['score'] = scores[:, 0]
    retro_result = reactants_df[reactants_df['prod_pred'] == target_product_smi]
    return retro_result


with open('data/candidates_single.txt') as f:
    candidates_smis = [s.rstrip() for s in f.readlines()]
n_candidates = len(candidates_smis)
candidates_smis = np.array(candidates_smis)

# NOTES: fix this part for multi-step multi-reactant reaction
target_reactant_smi = target_reactant_smi.split('.')

target_reactant_idx = list()
for smi_single in target_reactant_smi:
    idx_single = np.nonzero(candidates_smis == smi_single)[0][0]
    target_reactant_idx.append(idx_single)

target_reactant_idx = (tuple(sorted(target_reactant_idx)),)
# target_reactant_idx = ((target_reactant_idx,),)

results_path = Path('results')
results = list()
for r in results_path.iterdir():
    if r.stem.startswith('reaction{}_'.format(reaction_num)) and r.suffix == '.pickle':
        results.append(r)
if len(results) < 10:
    print('Experiments of reaction{} less than 10'.format(reaction_num),
          file=sys.stderr, flush=True)
else:
    results = results[:10]

summary = list()
summary_df = list()
for i, result in enumerate(results):
    with result.open('rb') as f:
        result = pickle.load(f)
    df = result_analyzer(reaction_num, result, target_reactant_idx, target_product_smi)
    summary_df.append(df)
    # Draw candidate reactant pairs
    # if len(df) > 0:
    #     df = df.sort_values(by='distance_true', axis=0, ascending=True).reset_index(drop=True)
    #     reactants_cand_fig = Path('results_figs') / 'reaction{}_cand'.format(reaction_num)
    #     os.makedirs(str(reactants_cand_fig), exist_ok=True)
    #     # FIXME: Only for single step
    #     reactants_cand = df['reactants'].apply(lambda x: x.idx2smi(candidates_smis)[0])
    #     reactants_cand_svg = draw_mols_smi(reactants_cand,
    #                                        legends=list(df['distance_true'].astype('str')))
    #     with open(str(reactants_cand_fig / 'experiments{}.svg'.format(i)), 'wt') as f:
    #         f.write(reactants_cand_svg)

os.makedirs('results_summary', exist_ok=True)
summary_path = Path('results_summary') / 'reaction{}.pickle'.format(reaction_num)
with summary_path.open('wb') as f:
    pickle.dump(summary_df, f, fix_imports=False)
