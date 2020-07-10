import os
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import shutil
import scipy.sparse as sp
from utils.fingerprint_utils import SparseFingerprintCsrMatrix
from utils.draw_utils import draw_mol_smi
from utils.draw_utils import draw_mols_smi

candidates_fps = sp.load_npz('data/candidates_fp_single.npz')
reaction_num = int(sys.argv[1])
current_file_path = Path(__file__)
results_path = current_file_path.absolute().parents[0] / 'results'
test = pd.read_pickle('data/preprocessed_liu_dataset/test_cls.pickle')
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


def result_analyzer(result, target_product_smi):
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

results = list()
for r in results_path.iterdir():
    if r.stem.startswith('reaction{}_'.format(reaction_num)) and r.suffix == '.pickle':
        results.append(r)
if len(results) < 10:
    print('Experiments of reaction{} less than 10'.format(reaction_num),
          file=sys.stderr, flush=True)
else:
    results = results[:10]

candidate_reactions = list()
candidate_reactions_fps = list()
for i, result in enumerate(results):
    with result.open('rb') as f:
        result = pickle.load(f)
    df = result_analyzer(result, target_product_smi)
    if len(df) > 0:
        candidate_reactants_fps = df['reactants'].apply(lambda x: x.idx2fp(candidates_fps))
        candidate_reactants_fps = np.concatenate(candidate_reactants_fps.values)
        candidate_reactants_fps = sp.csr_matrix(candidate_reactants_fps)
        product_fps = SparseFingerprintCsrMatrix(smis=[target_product_smi] * len(df)).tocsr()
        candidate_reactions_fps.append(sp.hstack([candidate_reactants_fps, product_fps], format='csc'))
    candidate_reactions.append(df)

summary_dir = Path('results_summary')
summary_dir.mkdir(exist_ok=True)
summary_fps_dir = summary_dir / 'candidate_reactions_fps'
summary_fps_dir.mkdir(exist_ok=True)
summary_path = summary_dir / 'reaction{}.pickle'.format(reaction_num)
with summary_path.open('wb') as f:
    pickle.dump(candidate_reactions, f, fix_imports=False)
if len(candidate_reactions_fps) > 0:
    candidate_reactions_fps = sp.vstack(candidate_reactions_fps, 'csc')
    sp.save_npz(str(summary_fps_dir / 'reaction{}'.format(reaction_num)), candidate_reactions_fps)
