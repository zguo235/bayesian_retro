#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


reaction_num = sys.argv[1]
candidate_df = pd.read_pickle('tsne_embedding/reaction{}_sorted.pickle'.format(reaction_num)).iloc[::-1, :]

fig_tsne, ax_tsne = plt.subplots()
fig_tsne.set_size_inches(12, 9)
ax_tsne.axis('equal')
norm = plt.Normalize(0, 1)
sm = plt.cm.ScalarMappable(cmap='Reds', norm=norm)
sm.set_array([])
sns.scatterplot(x='tsne_x', y='tsne_y', hue='prob_multi', data=candidate_df, palette='Reds', alpha='auto', ax=ax_tsne)
ax_tsne.set_xlabel('')
ax_tsne.set_ylabel('')
ax_tsne.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax_tsne.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
ax_tsne.get_legend().remove()
cbar = ax_tsne.figure.colorbar(sm)
cbar.ax.yaxis.set_ticks_position('right')
cbar.ax.tick_params(labelsize=24)
if candidate_df['true_reactant'].any():
    x_true, y_true = candidate_df[candidate_df['true_reactant']][['tsne_x', 'tsne_y']].values.flatten()
    ax_tsne.scatter([x_true], [y_true], marker='+', c='black', s=600, linewidth=6)
# ax_tsne.axis('off')
fig_tsne.savefig('tsne_embedding/reaction{}_tsne.png'.format(reaction_num))


from pandas.api.types import CategoricalDtype
import scipy.sparse as sp
from utils.ga_utils import csc_drop_zerocols
from pyclustering.cluster import xmeans

candidates_fps = sp.load_npz('data/candidates_fp_single.npz')
summary_fps = candidate_df['reactants_idx'].apply(lambda x: candidates_fps[np.concatenate(x)].sum(0))
summary_fps = sp.csc_matrix(np.concatenate(summary_fps.values, axis=0))
summary_fps_dropped = csc_drop_zerocols(summary_fps)
xmeans_init = xmeans.kmeans_plusplus_initializer(data=summary_fps_dropped.todense(), amount_centers=2)
initial_centers = xmeans_init.initialize()
xm = xmeans.xmeans(data=summary_fps_dropped.todense(), kmax=100, repeat=10)
xm.process()
clusters100 = xm.get_clusters()
centers100 = xm.get_centers()

candidate_df_split = list()
cluster_num = list()
for i, cluster in enumerate(clusters100):
    df_cluster = candidate_df.iloc[cluster]
    df_cluster['cluster'] = 'cluster ' + str(i+1)
    candidate_df_split.append(df_cluster)
    cluster_num.append('cluster ' + str(i+1))

summary_df_100clusters = pd.concat(candidate_df_split, axis=0)
cluster100_type = CategoricalDtype(categories=cluster_num, ordered=True)
summary_df_100clusters['cluster'] = summary_df_100clusters['cluster'].astype(cluster100_type)
summary_df_100clusters['size'] = False
summary_df_100clusters = summary_df_100clusters.sort_index(ascending=False)

fig_100clusters, ax_100clusters = plt.subplots()
fig_100clusters.set_size_inches(9, 9)
ax_100clusters.axis('equal')
scatter_100clusters = sns.scatterplot(x='tsne_x', y='tsne_y', hue='cluster',
                                      size='size', sizes={False: 20}, data=summary_df_100clusters,
                                      alpha='auto', ax=ax_100clusters, legend=False)
ax_100clusters.set_xlabel('')
ax_100clusters.set_ylabel('')
ax_100clusters.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax_100clusters.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
if summary_df_100clusters['true_reactant'].any():
    x_true, y_true = summary_df_100clusters[summary_df_100clusters['true_reactant']][['tsne_x', 'tsne_y']].values.flatten()
    ax_100clusters.scatter([x_true], [y_true], marker='+', c='black', s=600, linewidth=6)
# ax_100clusters.set_title('Number of clusters: {}'.format(len(summary_df_100clusters['cluster'].cat.categories)), fontsize=20)
fig_100clusters.savefig('tsne_embedding/reaction{}_100clusters.png'.format(reaction_num))
