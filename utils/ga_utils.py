import random
import sys
from time import time
import re
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy.linalg as LA
from typing import Union
from .fingerprint_utils import csc_drop_zerocols


class TargetProduct(object):

    def __init__(self, smi, similarity='tanimoto', verbose=False):
        self.smi = smi
        self.mol = Chem.MolFromSmiles(smi)
        self.fp = AllChem.GetMorganFingerprint(self.mol, radius=2, useFeatures=False)
        self.l2 = LA.norm(list(self.fp.GetNonzeroElements().values()))
        self.l1 = LA.norm(list(self.fp.GetNonzeroElements().values()), 1)
        self.similarity = similarity
        self.verbose = verbose

    def calc_ts(self, smi, distance=False):
        """ Calculate tanimoto similarity between target molecule and predicted product arrary.
        """

        try:
            mol = Chem.MolFromSmiles(smi)
            fp = AllChem.GetMorganFingerprint(mol, radius=2, useFeatures=False)
            sim = DataStructs.TanimotoSimilarity(self.fp, fp, returnDistance=distance)
        except Exception as e:
            if distance:
                sim = 1
            else:
                sim = 0
            if self.verbose:
                print('Original SMILES: {}'.format(smi), file=sys.stderr)
                # print(e, file=sys.stderr)
        return sim

    def calc_l2(self, smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            fp = self.fp - AllChem.GetMorganFingerprint(mol, radius=2, useFeatures=False)
            l2 = LA.norm(list(fp.GetNonzeroElements().values()))
        except Exception as e:
            # l2 = self.l2
            l2 = 9999
            if self.verbose:
                print('Original SMILES: {}'.format(smi), file=sys.stderr)
                # print(e, file=sys.stderr)
        return l2

    def calc_l1(self, smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            fp = self.fp - AllChem.GetMorganFingerprint(mol, radius=2, useFeatures=False)
            l1 = LA.norm(list(fp.GetNonzeroElements().values()), 1)
        except Exception as e:
            # l1 = self.l1
            l1 = 9999
            if self.verbose:
                print('Original SMILES: {}'.format(smi), file=sys.stderr)
                # print(e, file=sys.stderr)
        return l1

    def distance(self, products_array):
        products_distance = list()
        if self.similarity == 'tanimoto':
            for products_each_reaction in products_array:
                distance_each_reaction = \
                    [self.calc_ts(smi, distance=True) for smi in products_each_reaction]
                products_distance.append(distance_each_reaction)
        elif self.similarity == 'euclidean':
            for products_each_reaction in products_array:
                distance_each_reaction = [self.calc_l2(smi) for smi in products_each_reaction]
                products_distance.append(distance_each_reaction)
        elif self.similarity == 'manhattan':
            for products_each_reaction in products_array:
                distance_each_reaction = [self.calc_l1(smi) for smi in products_each_reaction]
                products_distance.append(distance_each_reaction)
        else:
            raise NotImplementedError
        return pd.DataFrame(products_distance)

    def likelihood(self, products_array, scores_array):
        products_sim = pd.DataFrame(products_array)
        if self.similarity == 'tanimoto':
            products_sim = products_sim.applymap(self.calc_ts)
        else:
            raise NotImplementedError
        scores_array = np.exp(scores_array)
        likelihood = products_sim * scores_array
        return likelihood.sum(axis=1)


pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex = re.compile(pattern)


class ParticalsTargetDistanceCalculator(object):

    def __init__(self, candidates_smis, predictor, target):
        self.candidates_smis = candidates_smis
        self.predictor = predictor
        self.target = target

    def prediction_smi(self, smis_list, **kwargs):
        smis_list_zipped = zip(*smis_list)
        product_previous_step = [""] * len(smis_list)
        top_score = np.zeros((len(smis_list), 1))
        for smi_list_step in smis_list_zipped:
            smi_list_step = zip(product_previous_step, smi_list_step)
            smi_list_step = [".".join(filter(None, smi)) for smi in smi_list_step]
            processed_smi = list()
            for s in smi_list_step:
                token = regex.findall(s)
                assert s == ''.join(s)
                processed_smi.append(' '.join(token))
            step_score, step_product = self.predictor.translate(src_data_iter=processed_smi, **kwargs)
            total_score = np.array(step_score) + top_score
            top_score = total_score[:, 0].reshape(len(total_score), 1)
            product_previous_step = [step_product[i][0] for i in range(len(smis_list))]
        return step_product, total_score

    def distance_index(self, reactants_list, **kwargs):
        smis_list = [r.idx2smi(self.candidates_smis) for r in reactants_list]
        products, scores = self.prediction_smi(smis_list, **kwargs)
        distances = self.target.distance(products)
        print(distances.mask(distances == 9999).describe())
        distances = distances.values
        distances_masked = np.ma.masked_values(distances, 9999)
        distances_likelihood = \
            loglikelihood_matrix_normalizer(scores, mask=distances_masked.mask)
        distances_adjusted = (distances_masked * distances_likelihood).sum(axis=1)
        distances_adjusted = distances_adjusted.filled(distances_adjusted.mean())
        return products, scores, distances_adjusted


def cal_random_weights(idx, method=None):
    if method is None:
        # return np.array([1] * len(idx), dtype=np.int8)
        return None
    else:
        indices = idx.ravel('C')
        data = [1] * (idx.shape[0] * idx.shape[1])
        indptr = [0]
        indptr.extend([idx.shape[1]] * idx.shape[0])
        indptr = np.cumsum(indptr)
        idx_sparse = csr_matrix((data, indices, indptr), shape=(len(idx), len(idx)), dtype=np.int8)
        transition_prob = idx_sparse.sum(axis=0)
        transition_prob = np.array(transition_prob)[0]
        if method == 'calib1':
            random_sampling_weights = 1 / transition_prob
            return random_sampling_weights
        elif method == 'calib2':
            transition_prob = transition_prob / transition_prob.sum()
            correction_term = 1 / len(idx) - transition_prob
            random_sampling_weights = 1 / len(idx) + np.where(correction_term > 0, correction_term, 0)
            return random_sampling_weights
        else:
            raise NotImplementedError


def loglikelihood_matrix_normalizer(array, mask):
    """Normalize loglikelihood matrix. Return likelihood matrix!
    Fix this function to normalize in log scale."""
    loglikelihood_matrix = np.ma.array(array, mask=mask)
    likelihood_matrix = np.exp(loglikelihood_matrix)
    norm = likelihood_matrix.sum(axis=1)
    return likelihood_matrix / norm[:, np.newaxis]


def reactant_random_sampling(n_reactants: int, n_particles: int,
                             n_candidates: int, candidates_prob: Union[list, None]) -> list:
    """Disabled replacement in sampling."""
    if candidates_prob is not None:
        assert n_candidates == len(candidates_prob)
    reactants_idx = list()
    for i in range(n_reactants):
        reactants_idx.append(np.random.choice(n_candidates, size=n_particles,
                                              replace=False, p=candidates_prob))
    reactants_idx = list(zip(*reactants_idx))
    return reactants_idx


def idx2smi(idx_list, candidates_smi):
    smi_list = list()
    for index in idx_list:
        smi = [candidates_smi[i] for i in index]
        smi = ".".join(smi)
        smi_list.append(smi)
    return smi_list


def idx2fp(idx_list, candidates_fp):
    fp_matrix = 0
    idx_list_unzipped = zip(*idx_list)
    for l in idx_list_unzipped:
        fp_matrix += candidates_fp[list(l)]
    return fp_matrix


class ReactantList(object):

    def __init__(self, reactant_num_list, n_candidates, exclude=list(),
                 weights=None, gibbs_index=[0, 0]):
        if weights is None:
            r_list = random.sample(range(n_candidates), sum(reactant_num_list))
            while set(exclude) & set(r_list):
                r_list = random.sample(range(n_candidates), sum(reactant_num_list))
        else:
            r_list = np.random.choice(n_candidates, size=sum(reactant_num_list), replace=False, p=weights)
        self.reactant_list = np.split(r_list, np.cumsum(reactant_num_list))[:-1]
        self.gibbs_index = gibbs_index
        self.immutable_list = self.immutable()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.immutable_list == other.immutable_list
        return False

    def __hash__(self):
        return hash(self.immutable_list)

    def immutable(self):
        return tuple(tuple(sorted(reactant)) for reactant in self.reactant_list)

    def idx2smi(self, candidates_smis):
        smi_list = list()
        for reactant_step in self.reactant_list:
            smi_step = [candidates_smis[index] for index in reactant_step]
            smi_step = ".".join(smi_step)
            smi_list.append(smi_step)
        return smi_list

    def idx2fp(self, candidates_fp):
        r_list = np.concatenate(self.reactant_list)
        fp = candidates_fp[r_list]
        return fp.sum(0)

    def nearest_neighbor(self, idx, exclude=list()):
        reactant_gibbs = self.reactant_list[self.gibbs_index[0]][self.gibbs_index[1]]
        reactant_next = idx[reactant_gibbs]
        reactant_next = set(reactant_next).difference(set(exclude))
        gibbs_index1 = (self.gibbs_index[1] + 1) % len(self.reactant_list[self.gibbs_index[0]])
        if (self.gibbs_index[1] + 1) // len(self.reactant_list[self.gibbs_index[0]]):
            gibbs_index0 = (self.gibbs_index[0] + 1) % len(self.reactant_list)
        else:
            gibbs_index0 = self.gibbs_index[0]
        new_reactantlist = deepcopy(self)
        new_reactantlist.gibbs_index = [gibbs_index0, gibbs_index1]
        nn_list = list()
        for i in reactant_next:
            nn = deepcopy(new_reactantlist)
            nn.reactant_list[self.gibbs_index[0]][self.gibbs_index[1]] = i
            nn.immutable_list = nn.immutable()
            nn_list.append(nn)
        return nn_list

    def random_sampling(self, n_candidates, exclude=list(), weights=None):
        if weights is None:
            reactant_next_random = random.choice(range(n_candidates))
            while reactant_next_random in exclude:
                reactant_next_random = random.choice(range(n_candidates))
        else:
            reactant_next_random = np.random.choice(n_candidates, p=weights)
        gibbs_index1 = (self.gibbs_index[1] + 1) % len(self.reactant_list[self.gibbs_index[0]])
        if (self.gibbs_index[1] + 1) // len(self.reactant_list[self.gibbs_index[0]]):
            gibbs_index0 = (self.gibbs_index[0] + 1) % len(self.reactant_list)
        else:
            gibbs_index0 = self.gibbs_index[0]
        new_reactantlist = deepcopy(self)
        new_reactantlist.gibbs_index = [gibbs_index0, gibbs_index1]
        new_reactantlist.reactant_list[self.gibbs_index[0]][self.gibbs_index[1]] = reactant_next_random
        new_reactantlist.immutable_list = new_reactantlist.immutable()
        return new_reactantlist

    def update_list(self, new_list, n_candidates, exclude=list(),
                    weights=None, gibbs_index=[0, 0]):
        old_list = list(map(len, self.reactant_list))
        reactant_list_new = list()
        if weights is None:
            new_reactant = random.sample(range(n_candidates), sum(new_list) - sum(old_list))
            while set(exclude) & set(new_reactant):
                new_reactant = random.sample(range(n_candidates), sum(new_list) - sum(old_list))
        else:
            new_reactant = np.random.choice(n_candidates, size=int(sum(new_list)-sum(old_list)),
                                            replace=False, p=weights)
        for i in range(len(new_list)):
            try:
                if new_list[i] > old_list[i]:
                    d = new_list[i] - old_list[i]
                    tmp = list(self.reactant_list[i])
                    tmp.extend(new_reactant[:d])
                    new_reactant = new_reactant[d:]
                    reactant_list_new.append(np.array(tmp))
                else:
                    reactant_list_new.append(self.reactant_list[i])
            except IndexError:
                d = new_list[i]
                reactant_list_new.append(np.array(new_reactant[:d]))
                new_reactant = new_reactant[d:]
        self.reactant_list = reactant_list_new
        self.gibbs_index = gibbs_index
        self.immutable_list = self.immutable()
        return self


def ga_clustering(reactants_candidates_fps, n_clusters, verbose=True):
    t0 = time()
    reactants_candidates_clustering_fps = csc_drop_zerocols(reactants_candidates_fps)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', n_init=10,
                             init_size=1000, batch_size=1000)
    kmeans.fit(reactants_candidates_clustering_fps)
    labels = kmeans.labels_
    if verbose:
        labels_uniq = np.unique(labels)
        n_clusters = len(labels_uniq)
        inertia = kmeans.inertia_
        print('cluster number:', n_clusters,
              '\ninertia:', inertia)
        print('Dropped fingerprint matrix shape:',
              reactants_candidates_clustering_fps.shape)
        print('KMeans fitting time: {:.3f}s'.format(time() - t0))
    return labels


def group_uniform_sampling(reactants_candidates_df, top=100):
    df = reactants_candidates_df.sort_values('weights', ascending=False)
    grouped = df.groupby('labels')
    reactants_candidates_uniform = pd.DataFrame()
    i = 0
    while len(reactants_candidates_uniform) < top:
        step_list = list()
        for name, group in grouped:
            try:
                step_list.append(group.iloc[i])
            except IndexError:
                pass
        step_df = pd.DataFrame(step_list)
        reactants_candidates_uniform = pd.concat([reactants_candidates_uniform,
                                                  step_df.sort_values('weights', ascending=False)])
        i += 1
    return reactants_candidates_uniform[:top][['reactants', 'labels', 'distance_pred']]


def group_weight_sampling(reactants_candidates_df, size):
    df = reactants_candidates_df.sort_values('weights', ascending=False)
    grouped = df.groupby('labels')
    group_weight = grouped['weights'].mean()
    group_weight = group_weight / group_weight.sum()
    group_sampling_size = np.ceil(group_weight*size).astype(int)

    def ordering(x):
        x['sampling_size'] = group_sampling_size[x['labels'].iloc[0]]
        x['order'] = list(range(len(x)))
        return x
    df = grouped.apply(ordering)
    df['proposal'] = df.apply(lambda x: x['order'] < x['sampling_size'], axis=1)
    if df['proposal'].sum() < size:
        n_reple = size - df['proposal'].sum()
        reple_index = df[~df['proposal']].index
        reple_index = reple_index[:n_reple]
        df.loc[reple_index, 'proposal'] = True
    return df[df['proposal']][['reactants', 'labels', 'distance_pred']]


def distance2weights(distance, temperature):
    distance = np.where(distance > 0, distance, 0)
    return np.exp(-distance/temperature)
