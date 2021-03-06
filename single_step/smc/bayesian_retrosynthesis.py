import os
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
os.environ["NUMEXPR_NUM_THREADS"] = "10"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
from time import time
import pickle
import random
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from utils.surrogate_model_utils import test_model
from utils.transformer_utils import build_translator
from utils.ga_utils import TargetProduct, ParticalsTargetDistanceCalculator
from utils.ga_utils import ReactantList, ga_clustering
from utils.ga_utils import group_uniform_sampling, group_weight_sampling
from utils.ga_utils import distance2weights, cal_random_weights
from utils.draw_utils import clustering_fig
import torch
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
# import warnings
# warnings.simplefilter("error", category=UserWarning)


# Preamble
# ----------------------------------------------------------------------------------------------------------------------
# Hyperparameter
n_steps = [100, 600]
n_particles = 1000
random_sampling = 0.5
tm_size = 100
try:
    savedir = sys.argv[2]
except IndexError:
    savedir = "time_test"
online = 'off'
# Target reaction
with open('data/preprocessed_liu_dataset/test_sampled.pickle', 'rb') as f:
    test_reaction = pickle.load(f)[:100]
reaction_num = int(sys.argv[1])
# reaction_num = 0
original_index = test_reaction.index[reaction_num]
target_reactant_smi, target_product_smi = test_reaction.loc[original_index, ['reactant', 'product']]
target = TargetProduct(target_product_smi, similarity='euclidean')
# Load transition matrix
idx = np.load('data/idx_single.npy')[:, :tm_size]
prob = np.load('data/prob_single.npy')[:, :tm_size]
random_sampling_weights = cal_random_weights(idx, method='calib1')
# Load candidates
with open('data/candidates_single.txt') as f:
    candidates_smis = [s.strip() for s in f.readlines()]
n_candidates = len(candidates_smis)
candidates_smis = np.array(candidates_smis)
candidates_fp = scipy.sparse.load_npz('data/candidates_fp_single.npz')
if target_product_smi in candidates_smis:
    target_product_idx = np.argwhere(candidates_smis == target_product_smi)[0][0]
    random_sampling_weights[target_product_idx] = 0
    random_sampling_weights = random_sampling_weights / random_sampling_weights.sum()
else:
    target_product_idx = None
    random_sampling_weights = random_sampling_weights / random_sampling_weights.sum()
# Build translator model
use_gpu = torch.cuda.is_available()
translator = build_translator(use_gpu=use_gpu)
# ----------------------------------------------------------------------------------------------------------------------

# Train surrogate model
# ----------------------------------------------------------------------------------------------------------------------
print('Training surrogate model ...')
train_X = scipy.sparse.load_npz('data/preprocessed_liu_dataset/train_X.npz')
train_y_smi = list()
with open('data/preprocessed_liu_dataset/tgt-train.txt', 'rt') as f:
    for smi in f.readlines():
        train_y_smi.append([smi.strip()])
train_y = target.distance(train_y_smi).values.squeeze()
valid_X = scipy.sparse.load_npz('data/preprocessed_liu_dataset/valid_X.npz')
valid_y_smi = list()
with open('data/preprocessed_liu_dataset/tgt-valid.txt', 'rt') as f:
    for smi in f.readlines():
        valid_y_smi.append([smi.strip()])
valid_y = target.distance(valid_y_smi).values.squeeze()
lgb_train = lgb.Dataset(train_X, train_y)
lgb_eval = lgb.Dataset(valid_X, valid_y, reference=lgb_train)

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['l1', 'l2'],
    'num_leaves': 63,
    'learning_rate': 0.1,
    'min_child_samples': 10,
    'subsample': 1,
    'colsample_bytree': 0.8,
    'verbose': 0
}
regr = lgb.train(lgb_params, lgb_train, num_boost_round=500, valid_sets=lgb_eval,
                 early_stopping_rounds=5, verbose_eval=10)

test_X = scipy.sparse.load_npz('data/preprocessed_liu_dataset/test_X.npz')
test_y_smi = list()
with open('data/preprocessed_liu_dataset/tgt-test.txt', 'rt') as f:
    for smi in f.readlines():
        test_y_smi.append([smi.strip()])
test_y = target.distance(test_y_smi).values.squeeze()
fig_dir = os.path.join('figures', savedir)
os.makedirs(fig_dir, exist_ok=True)
save_path = os.path.join('results', savedir)
os.makedirs(save_path, exist_ok=True)
surrogate_model_file = 'regr.pkl'
surrogate_model_path = os.path.join(save_path, surrogate_model_file)
with open(surrogate_model_path, 'wb') as f:
    pickle.dump(regr, f)
fig_lgb = 'LightGBM_step{:0>4}.png'.format(0)
fig_path = os.path.join(fig_dir, fig_lgb)
test_model(regr, (test_X, test_y), mean_squared_error,
           'LightGBM', fig_path=fig_path, target_idx=original_index)
# ----------------------------------------------------------------------------------------------------------------------

# First loop
# ----------------------------------------------------------------------------------------------------------------------
reactant_num_list = [1]
print("Start first loop...", flush=True)
step = 0
t_start = time()
reactants_list = [ReactantList(reactant_num_list, n_candidates, exclude=[target_product_idx]) for _ in range(n_particles)]
ga_idx = set(reactants_list)
distance_calculator = ParticalsTargetDistanceCalculator(candidates_smis, translator, target)
products, scores, distance_adjusted = \
    distance_calculator.distance_index(reactants_list, batch_size=100, attn_debug=False)

if savedir == "time_test":
    result = list()
    result_step = ((reactants_list, distance_adjusted), products, scores)
    result.append(result_step)
    elapsed_time = time() - t_start
    print('Finished step {},'.format(step), 'elapsed time: {:.2f}.'.format(elapsed_time),
          file=sys.stdout, flush=True)
    print()
else:
    reactants_fps = [r.idx2fp(candidates_fp) for r in reactants_list]
    reactants_fps = scipy.sparse.csc_matrix(np.concatenate(reactants_fps, axis=0))
    reactants_distance = regr.predict(reactants_fps)
    reactants_df = pd.DataFrame(reactants_list, columns=['reactants'])
    reactants_df['labels'] = 'random'
    reactants_df['distance_pred'] = reactants_distance
    reactants_df['distance_true'] = distance_adjusted
    result_step = (reactants_df, products, scores)
    result = list()
    result.append(result_step)
    file_name = 'step_' + str(step) + '.pickle'
    file_path = os.path.join(save_path, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(result_step, f)
    # del reactants_fps, reactants_distance, reactants_df, products, scores, result_step
    elapsed_time = time() - t_start
    print('Finished step {},'.format(step), 'elapsed time: {:.2f}.'.format(elapsed_time),
          file=sys.stdout, flush=True)
    print()
# ----------------------------------------------------------------------------------------------------------------------

# One reactant sme loop
# ----------------------------------------------------------------------------------------------------------------------
while step < n_steps[0]:
    t_start = time()
    step += 1
    temperature = 20 / (step % 100 + 0.1)
    temperature = np.max((temperature, 1))
    # Nearest neighbor
    reactants_tops_idx = distance_adjusted.argsort()[:int(n_particles / 10)]
    reactants_tops = [reactants_list[i] for i in reactants_tops_idx]
    reactants_candidates = list()
    for r in reactants_tops:
        reactants_candidates.extend(r.nearest_neighbor(idx, exclude=[target_product_idx]))
    reactants_candidates = list(set(reactants_candidates) - ga_idx)
    # Calculate weight using surrogate model
    reactants_candidates_fps = [r.idx2fp(candidates_fp) for r in reactants_candidates]
    reactants_candidates_fps = scipy.sparse.csc_matrix(np.concatenate(reactants_candidates_fps, axis=0))
    distance = regr.predict(reactants_candidates_fps)
    weights = distance2weights(distance, temperature)
    # Clustering
    n_clusters = np.random.randint(20, 50)
    if savedir == 'time_test':
        labels = ga_clustering(reactants_candidates_fps, n_clusters, verbose=False)
    else:
        labels = ga_clustering(reactants_candidates_fps, n_clusters)
    # Proposal for transformer
    n_proposal = int(n_particles * (1 - random_sampling))
    df = pd.DataFrame()
    df['reactants'] = reactants_candidates
    df['labels'] = labels
    df['distance_pred'] = distance
    df['weights'] = weights
    reactants_proposal_uniform = group_uniform_sampling(df, top=100)
    df['proposal'] = False
    df.loc[reactants_proposal_uniform.index, 'proposal'] = True
    reactants_proposal_weight = group_weight_sampling(df[~df['proposal']], size=(n_proposal-100))
    reactants_proposal = pd.concat([reactants_proposal_uniform, reactants_proposal_weight])
    # Random-sampled particals for transformer
    reactants_list = list(reactants_proposal['reactants'])
    ga_idx.update(reactants_list)
    reactants_random_gibbs = set()
    while len(reactants_random_gibbs) < len(reactants_tops)*2:
        reactants_random_gibbs.update([r.random_sampling(n_candidates, exclude=[target_product_idx],
                                                         weights=random_sampling_weights) for r in reactants_tops])
        reactants_random_gibbs = reactants_random_gibbs - ga_idx
    reactants_random_gibbs = list(reactants_random_gibbs)
    reactants_list.extend(reactants_random_gibbs)
    ga_idx.update(reactants_random_gibbs)
    reactants_random_joint = set()
    while len(reactants_random_joint) < (n_particles - len(reactants_list)):
        reactants_random_joint.update([ReactantList(reactant_num_list, n_candidates, exclude=[target_product_idx],
                                                    weights=random_sampling_weights) for _ in range(n_particles//2)])
        reactants_random_joint = reactants_random_joint - ga_idx
    reactants_random_joint = random.sample(reactants_random_joint, n_particles-len(reactants_list))
    reactants_list.extend(reactants_random_joint)
    ga_idx.update(reactants_random_joint)
    # Transformer prediction
    products, scores, distance_adjusted = \
        distance_calculator.distance_index(reactants_list, batch_size=100, attn_debug=False)

    if online == 'on':
        reactants_fps = [r.idx2fp(candidates_fp) for r in reactants_list]
        reactants_fps = scipy.sparse.csc_matrix(np.concatenate(reactants_fps, axis=0))
        online_X = reactants_fps
        online_y = distance_adjusted
        train_X, valid_X, train_y, valid_y = train_test_split(online_X, online_y, test_size=0.2)
        lgb_train = lgb.Dataset(train_X, train_y)
        lgb_eval = lgb.Dataset(valid_X, valid_y, reference=lgb_train)
        if savedir == 'time_test':
            regr = lgb.train(lgb_params, lgb_train, num_boost_round=10, init_model=regr,
                             valid_sets=lgb_eval, early_stopping_rounds=5, verbose_eval=False)
        else:
            regr = lgb.train(lgb_params, lgb_train, num_boost_round=10, init_model=regr,
                             valid_sets=lgb_eval, early_stopping_rounds=5)
            reactants_random = reactants_random_gibbs + reactants_random_joint
            reactants_random_fps = reactants_fps.tocsr()[len(reactants_proposal):]
            reactants_random_distance = regr.predict(reactants_random_fps)
            fig_lgb = 'LightGBM_step{:0>4}.png'.format(step)
            fig_path = os.path.join(fig_dir, fig_lgb)
            test_model(regr, (test_X, test_y), mean_squared_error,
                       'LightGBM', fig_path=fig_path, target_idx=original_index)
    else:
        if savedir == 'time_test':
            pass
        else:
            reactants_random = reactants_random_gibbs + reactants_random_joint
            reactants_random_fps = [r.idx2fp(candidates_fp) for r in reactants_random]
            reactants_random_fps = scipy.sparse.csc_matrix(np.concatenate(reactants_random_fps, axis=0))
            reactants_random_distance = regr.predict(reactants_random_fps)

    if savedir == 'time_test':
        result_step = ((reactants_list, distance_adjusted), products, scores)
        result.append(result_step)
        elapsed_time = time() - t_start
        print('Finished step {},'.format(step), 'elapsed time: {:.2f}.'.format(elapsed_time),
              file=sys.stdout, flush=True)
        print()
    else:
        reactants_random = pd.DataFrame(reactants_random, columns=['reactants'])
        reactants_random['labels'] = 'random'
        reactants_random['distance_pred'] = reactants_random_distance
        reactants_df = pd.concat([reactants_proposal, reactants_random], ignore_index=True)
        reactants_df['distance_true'] = distance_adjusted
        result_step = (reactants_df, products, scores)
        result.append(result_step)
        # Figure of clustering
        cluster_fig = 'cluster_step{:0>4}'.format(step)
        cluster_fig = os.path.join(fig_dir, cluster_fig)
        clustering_fig(df, "Clustering step_{}".format(step), cluster_fig)
        # Result of this step
        file_name = 'step_' + str(step) + '.pickle'
        file_path = os.path.join(save_path, file_name)
        with open(file_path, 'wb') as f:
            pickle.dump(result_step, f)
        elapsed_time = time() - t_start
        print('Finished step {},'.format(step), 'elapsed time: {:.2f}.'.format(elapsed_time),
              file=sys.stdout, flush=True)
        print()
# ----------------------------------------------------------------------------------------------------------------------

# Update reactant_num_list
# ----------------------------------------------------------------------------------------------------------------------
print("Update reactant from 1 step 1 reactant to 1 step 2 reactants")
reactant_num_list = [2]
print("Reload surrogate model...")
with open(surrogate_model_path, 'rb') as f:
    regr = pickle.load(f)
t_start = time()
step += 1
# step = 0
reactants_list = [r.update_list(reactant_num_list, n_candidates, exclude=[target_product_idx]) for r in reactants_list]
ga_idx = set(reactants_list)
# ga_idx.update(reactants_list)
products, scores, distance_adjusted = \
    distance_calculator.distance_index(reactants_list, batch_size=100, attn_debug=False)

if savedir == "time_test":
    # result = list()
    result_step = ((reactants_list, distance_adjusted), products, scores)
    result.append(result_step)
    elapsed_time = time() - t_start
    print('Finished step {}(update step),'.format(step), 'elapsed time: {:.2f}.'.format(elapsed_time),
          file=sys.stdout, flush=True)
    print()
else:
    reactants_fps = [r.idx2fp(candidates_fp) for r in reactants_list]
    reactants_fps = scipy.sparse.csc_matrix(np.concatenate(reactants_fps, axis=0))
    reactants_distance = regr.predict(reactants_fps)
    reactants_df = pd.DataFrame(reactants_list, columns=['reactants'])
    reactants_df['labels'] = 'updated'
    reactants_df['distance_pred'] = reactants_distance
    reactants_df['distance_true'] = distance_adjusted
    result_step = (reactants_df, products, scores)
    # result = list()
    result.append(result_step)
    file_name = 'step_' + str(step) + '.pickle'
    file_path = os.path.join(save_path, file_name)
    with open(file_path, 'wb') as f:
        pickle.dump(result_step, f)
    # del reactants_fps, reactants_distance, reactants_df, products, scores, result_step
    elapsed_time = time() - t_start
    print('Finished step {}(update step),'.format(step), 'elapsed time: {:.2f}.'.format(elapsed_time),
          file=sys.stdout, flush=True)
    print()
# ----------------------------------------------------------------------------------------------------------------------

# Two reactant sme loop
# ----------------------------------------------------------------------------------------------------------------------
while step < n_steps[1]:
    t_start = time()
    step += 1
    temperature = 20 / (step % 100 + 0.1)
    temperature = np.max((temperature, 1))
    # Nearest neighbor
    reactants_tops_idx = distance_adjusted.argsort()[:int(n_particles / 10)]
    reactants_tops = [reactants_list[i] for i in reactants_tops_idx]
    reactants_candidates = list()
    for r in reactants_tops:
        reactants_candidates.extend(r.nearest_neighbor(idx, exclude=[target_product_idx]))
    reactants_candidates = list(set(reactants_candidates) - ga_idx)
    # Calculate weight using surrogate model
    reactants_candidates_fps = [r.idx2fp(candidates_fp) for r in reactants_candidates]
    reactants_candidates_fps = scipy.sparse.csc_matrix(np.concatenate(reactants_candidates_fps, axis=0))
    distance = regr.predict(reactants_candidates_fps)
    weights = distance2weights(distance, temperature)
    # Clustering
    n_clusters = np.random.randint(20, 50)
    if savedir == 'time_test':
        labels = ga_clustering(reactants_candidates_fps, n_clusters, verbose=False)
    else:
        labels = ga_clustering(reactants_candidates_fps, n_clusters)
    # Proposal for transformer
    n_proposal = int(n_particles * (1 - random_sampling))
    df = pd.DataFrame()
    df['reactants'] = reactants_candidates
    df['labels'] = labels
    df['distance_pred'] = distance
    df['weights'] = weights
    reactants_proposal_uniform = group_uniform_sampling(df, top=100)
    df['proposal'] = False
    df.loc[reactants_proposal_uniform.index, 'proposal'] = True
    reactants_proposal_weight = group_weight_sampling(df[~df['proposal']], size=(n_proposal-100))
    reactants_proposal = pd.concat([reactants_proposal_uniform, reactants_proposal_weight])
    # Random-sampled particals for transformer
    reactants_list = list(reactants_proposal['reactants'])
    ga_idx.update(reactants_list)
    reactants_random_gibbs = set()
    while len(reactants_random_gibbs) < len(reactants_tops)*2:
        reactants_random_gibbs.update([r.random_sampling(n_candidates, exclude=[target_product_idx],
                                                         weights=random_sampling_weights) for r in reactants_tops])
        reactants_random_gibbs = reactants_random_gibbs - ga_idx
    reactants_random_gibbs = list(reactants_random_gibbs)
    reactants_list.extend(reactants_random_gibbs)
    ga_idx.update(reactants_random_gibbs)
    reactants_random_joint = set()
    while len(reactants_random_joint) < (n_particles - len(reactants_list)):
        reactants_random_joint.update([ReactantList(reactant_num_list, n_candidates, exclude=[target_product_idx],
                                                    weights=random_sampling_weights) for _ in range(n_particles//2)])
        reactants_random_joint = reactants_random_joint - ga_idx
    reactants_random_joint = random.sample(reactants_random_joint, n_particles-len(reactants_list))
    reactants_list.extend(reactants_random_joint)
    ga_idx.update(reactants_random_joint)
    # Transformer prediction
    products, scores, distance_adjusted = \
        distance_calculator.distance_index(reactants_list, batch_size=100, attn_debug=False)

    if online == 'on':
        reactants_fps = [r.idx2fp(candidates_fp) for r in reactants_list]
        reactants_fps = scipy.sparse.csc_matrix(np.concatenate(reactants_fps, axis=0))
        online_X = reactants_fps
        online_y = distance_adjusted
        train_X, valid_X, train_y, valid_y = train_test_split(online_X, online_y, test_size=0.2)
        lgb_train = lgb.Dataset(train_X, train_y)
        lgb_eval = lgb.Dataset(valid_X, valid_y, reference=lgb_train)
        if savedir == 'time_test':
            regr = lgb.train(lgb_params, lgb_train, num_boost_round=10, init_model=regr,
                             valid_sets=lgb_eval, early_stopping_rounds=5, verbose_eval=False)
        else:
            regr = lgb.train(lgb_params, lgb_train, num_boost_round=10, init_model=regr,
                             valid_sets=lgb_eval, early_stopping_rounds=5)
            reactants_random = reactants_random_gibbs + reactants_random_joint
            reactants_random_fps = reactants_fps.tocsr()[len(reactants_proposal):]
            reactants_random_distance = regr.predict(reactants_random_fps)
            fig_lgb = 'LightGBM_step{:0>4}.png'.format(step)
            fig_path = os.path.join(fig_dir, fig_lgb)
            test_model(regr, (test_X, test_y), mean_squared_error,
                       'LightGBM', fig_path=fig_path, target_idx=original_index)
    else:
        if savedir == 'time_test':
            pass
        else:
            reactants_random = reactants_random_gibbs + reactants_random_joint
            reactants_random_fps = [r.idx2fp(candidates_fp) for r in reactants_random]
            reactants_random_fps = scipy.sparse.csc_matrix(np.concatenate(reactants_random_fps, axis=0))
            reactants_random_distance = regr.predict(reactants_random_fps)

    if savedir == 'time_test':
        result_step = ((reactants_list, distance_adjusted), products, scores)
        result.append(result_step)
        elapsed_time = time() - t_start
        print('Finished step {},'.format(step), 'elapsed time: {:.2f}.'.format(elapsed_time),
              file=sys.stdout, flush=True)
        print()
    else:
        reactants_random = pd.DataFrame(reactants_random, columns=['reactants'])
        reactants_random['labels'] = 'random'
        reactants_random['distance_pred'] = reactants_random_distance
        reactants_df = pd.concat([reactants_proposal, reactants_random], ignore_index=True)
        reactants_df['distance_true'] = distance_adjusted
        result_step = (reactants_df, products, scores)
        result.append(result_step)
        # Figure of clustering
        cluster_fig = 'cluster_step{:0>4}'.format(step)
        cluster_fig = os.path.join(fig_dir, cluster_fig)
        clustering_fig(df, "Clustering step_{}".format(step), cluster_fig)
        # Result of this step
        file_name = 'step_' + str(step) + '.pickle'
        file_path = os.path.join(save_path, file_name)
        with open(file_path, 'wb') as f:
            pickle.dump(result_step, f)
        elapsed_time = time() - t_start
        print('Finished step {},'.format(step), 'elapsed time: {:.2f}.'.format(elapsed_time),
              file=sys.stdout, flush=True)
        print()
# ----------------------------------------------------------------------------------------------------------------------

if savedir == 'time_test':
    result_file = os.path.join('results', 'reaction' + str(reaction_num) + '.pickle')
    with open(result_file, 'wb') as f:
        pickle.dump(result, f)
else:
    result_file = os.path.join('results', savedir + '.pickle')
    with open(result_file, 'wb') as f:
        pickle.dump(result, f)

print('Finished reaction No.{},'.format(reaction_num), 'Product Smiles:', target_product_smi,
      file=sys.stdout, flush=True)
