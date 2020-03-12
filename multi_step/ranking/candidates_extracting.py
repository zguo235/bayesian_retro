import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import shutil
import scipy.sparse as sp
import torch
from rdkit import Chem
from utils.transformer_utils import build_translator
from utils.fingerprint_utils import SparseFingerprintCsrMatrix
from utils.draw_utils import draw_mol_smi
from utils.draw_utils import draw_mols_smi


# Build translator
use_gpu = torch.cuda.is_available()
translator = build_translator(use_gpu=use_gpu)
# Regular expression for SMILES used in Molecular Transformer
import re
pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex = re.compile(pattern)
# Import candidate reactant SMILES and fingerprints
# import scipy
with open('data/candidates_single.txt') as f:
    candidates_smis = [s.strip() for s in f.readlines()]
n_candidates = len(candidates_smis)
candidates_smis = np.array(candidates_smis)
candidates_fps = sp.load_npz('data/candidates_fp_single.npz')
reactant_num_list = [2, 1]
reaction_num = int(sys.argv[1])
test_2steps = pd.read_pickle('data/preprocessed_liu_dataset/test_2steps.pickle')
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


# Draw reactant and product
os.makedirs('results_figs', exist_ok=True)
target_reactant_smi_fig = ['.'.join(s) for s in target_reactant_smi_list]
target_reactant_svg = draw_mols_smi(target_reactant_smi_fig)
target_product_svg = draw_mol_smi(target_product_smi)
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

    reactants_df['prod_pred'] = products[:, 0]
    reactants_df['score'] = scores[:, 0]
    retro_result = reactants_df[reactants_df['prod_pred'] == target_product_smi]
    return retro_result


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

candidate_reactions = list()
candidate_reactions_step1_fps = list()
candidate_reactions_step2_fps = list()
candidate_reactions_len = list()
for i, result in enumerate(results):
    with result.open('rb') as f:
        result = pickle.load(f)
    df = result_analyzer(result, target_product_smi)
    candidate_reactions_len.append(len(df))
    if len(df) > 0:
        # Product in each step
        reaction_smis = df['reactants'].apply(lambda x: x.idx2smi(candidates_smis))
        smis_list_zipped = zip(*list(reaction_smis))
        product_previous_step = [""] * len(reaction_smis)
        j = 1
        for smi_list_step in smis_list_zipped:
            smi_list_step = zip(product_previous_step, smi_list_step)
            smi_list_step = [".".join(filter(None, smi)) for smi in smi_list_step]
            df['reactant_smi_step{}'.format(j)] = smi_list_step
            processed_smis = list()
            for s in smi_list_step:
                token = regex.findall(s)
                assert s == ''.join(token)
                processed_smis.append(' '.join(token))
            step_score, step_product = translator.translate(src_data_iter=processed_smis, batch_size=100, attn_debug=False)
            product_previous_step = [step_product[n][0] for n in range(len(reaction_smis))]
            df['product_smi_step{}'.format(j)] = product_previous_step
            j += 1
        product_step_valid = [False if Chem.MolFromSmiles(smi) is None else True for smi in df['product_smi_step1']]
        df = df[product_step_valid]

        candidate_reactants_step1_fps = SparseFingerprintCsrMatrix(smis=df['reactant_smi_step1']).tocsr()
        candidate_products_step1_fps = SparseFingerprintCsrMatrix(smis=df['product_smi_step1']).tocsr()
        candidate_reactions_step1_fps.append(sp.hstack([candidate_reactants_step1_fps, candidate_products_step1_fps], format='csc'))
        candidate_reactants_step2_fps = SparseFingerprintCsrMatrix(smis=df['reactant_smi_step2']).tocsr()
        candidate_products_step2_fps = SparseFingerprintCsrMatrix(smis=df['product_smi_step2']).tocsr()
        candidate_reactions_step2_fps.append(sp.hstack([candidate_reactants_step2_fps, candidate_products_step2_fps], format='csc'))
        candidate_reactions.append(df)

summary_dir = Path('results_summary')
summary_dir.mkdir(exist_ok=True)
summary_fps_dir = summary_dir / 'candidate_reactions_fps'
summary_fps_dir.mkdir(exist_ok=True)
summary_path = summary_dir / 'reaction{}.pickle'.format(reaction_num)
with summary_path.open('wb') as f:
    pickle.dump((candidate_reactions, candidate_reactions_len), f, fix_imports=False)
if len(candidate_reactions) > 0:
    candidate_reactions = pd.concat(candidate_reactions, axis=0, ignore_index=True)
    candidate_reactions_step1_fps = sp.vstack(candidate_reactions_step1_fps, 'csc')
    sp.save_npz(os.path.join('results_summary', 'candidate_reactions_fps', 'reaction{}_step1'.format(reaction_num)), candidate_reactions_step1_fps)
    candidate_reactions_step2_fps = sp.vstack(candidate_reactions_step2_fps, 'csc')
    sp.save_npz(os.path.join('results_summary', 'candidate_reactions_fps', 'reaction{}_step2'.format(reaction_num)), candidate_reactions_step2_fps)
