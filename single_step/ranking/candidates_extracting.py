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
test = pd.read_pickle('data/preprocessed_liu_dataset/test_sampled.pickle')
target_reactant_smi, target_product_smi = test.iloc[reaction_num, [0, 1]]
os.makedirs('results_figs', exist_ok=True)

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


def result_analyzer(reaction_num, result, target_reactant_idx, target_product_smi):
    reactants_df, products, scores = zip(*result)
    products = [np.array(p) for p in products]
    scores = [np.array(s) for s in scores]
    reactants_df = pd.concat(reactants_df, ignore_index=True)
    products = np.concatenate(products)
    scores = np.concatenate(scores)

    identical_m = products == target_product_smi
    identical_any = np.any(identical_m, axis=1)
    reactants_found = reactants_df[identical_any]
    products_found = products[identical_any]
    scores_found = scores[identical_any]
    identical_found = identical_m[identical_any]

    reactants_found['identical_prod_id'] = np.nonzero(identical_found)[1]
    reactants_found['score'] = np.sum(identical_found * scores_found, axis=1)

    products_found = pd.DataFrame(products_found, columns=['prod_pred{}'.format(i) for i in range(5)])
    scores_found = pd.DataFrame(scores_found, columns=['score{}'.format(i) for i in range(5)])
    retro_result = pd.concat([reactants_found.reset_index(), products_found, scores_found], axis=1)
    return retro_result.set_index('index')


with open('data/candidates_single.txt') as f:
    candidates_smis = [s.rstrip() for s in f.readlines()]
n_candidates = len(candidates_smis)
candidates_smis = np.array(candidates_smis)

target_reactant_smi = target_reactant_smi.split('.')

target_reactant_idx = list()
for smi_single in target_reactant_smi:
    idx_single = np.nonzero(candidates_smis == smi_single)[0][0]
    target_reactant_idx.append(idx_single)

target_reactant_idx = (tuple(sorted(target_reactant_idx)),)

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

os.makedirs('results_summary', exist_ok=True)
summary_path = Path('results_summary') / 'reaction{}.pickle'.format(reaction_num)
with summary_path.open('wb') as f:
    pickle.dump(summary_df, f, fix_imports=False)
