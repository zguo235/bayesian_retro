import sys
import re
from copy import deepcopy
import numpy as np
import pandas as pd
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
        for smi_list_step in smis_list_zipped:
            smi_list_step = zip(product_previous_step, smi_list_step)
            smi_list_step = [".".join(filter(None, smi)) for smi in smi_list_step]
            processed_smi = list()
            for s in smi_list_step:
                token = regex.findall(s)
                assert s == ''.join(s)
                processed_smi.append(' '.join(token))
            step_score, step_product = self.predictor.translate(src_data_iter=processed_smi, **kwargs)
            product_previous_step = [step_product[i][0] for i in range(len(smis_list))]
            return step_product, step_score

    # def prediction_index(self, reactants_index, **kwargs):
    #     reactants_smi = list()
    #     for index in reactants_index:
    #         reactant_smi = ''
    #         for i in index:
    #             reactant_smi += self.candidates_smis[i] + '.'
    #         reactants_smi.append(reactant_smi[:-1])
    #     scores, products = self.prediction_smi(reactants_smi, **kwargs)
    #     return products, scores

    def distance_index(self, reactants_list, **kwargs):
        smis_list = [r.idx2smi(self.candidates_smis) for r in reactants_list]
        products, scores = self.prediction_smi(smis_list, **kwargs)
        # products, scores = self.prediction_index(reactants_list, **kwargs)
        distances = self.target.distance(products)
        print(distances.mask(distances == 9999).describe())
        distances = distances.values
        distances_masked = np.ma.masked_values(distances, 9999)
        distances_likelihood = \
            loglikelihood_matrix_normalizer(scores, mask=distances_masked.mask)
        distances_adjucted = (distances_masked * distances_likelihood).sum(axis=1)
        distances_adjusted = distances_adjucted.filled(distances_adjucted.mean())
        return products, scores, distances_adjusted


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

    def __init__(self, reactant_num_list, n_candidates, candidates_prob, gibbs_index=[0, 0]):
        r_list = np.random.choice(n_candidates, sum(reactant_num_list), replace=False, p=candidates_prob)
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

    def nearest_neighbor(self, idx, exclude=None):
        reactant_gibbs = self.reactant_list[self.gibbs_index[0]][self.gibbs_index[1]]
        reactant_next = idx[reactant_gibbs]
        if exclude in reactant_next:
            reactant_next = list(reactant_next)
            reactant_next.remove(exclude)
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


# def test_ReactantList(reactant_num_list):
#     test_reactantlist = ReactantList(reactant_num_list, n_candidates, candidates_prob)
#     # global test_reactantlist
#     print("SMILES:", test_reactantlist.idx2smi(candidates_smis))
#     print("Fingerprint:", test_reactantlist.idx2fp(candidates_fp))
#     print("Nearest neighbor:")
#     for t in test_reactantlist.nearest_neighbor(idx)[:10]:
#         print(t.__dict__)


def ga_clustering(reactants_candidates_fps, n_clusters):
    reactants_candidates_clustering_fps = csc_drop_zerocols(reactants_candidates_fps)
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', n_init=10,
                             init_size=1000, batch_size=1000)
    kmeans.fit(reactants_candidates_clustering_fps)
    labels = kmeans.labels_
    labels_uniq = np.unique(labels)
    n_clusters = len(labels_uniq)
    inertia = kmeans.inertia_
    print('cluster number:', n_clusters,
          '\ninertia:', inertia)
    print('Dropped fingerprint matrix shape:',
          reactants_candidates_clustering_fps.shape)
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
