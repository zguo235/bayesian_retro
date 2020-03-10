import os
import sys
from time import time
import numpy as np
from lvdmaaten import bhtsne


perplexity = 50
theta = 0.5
directory = os.getcwd()
fpname = sys.argv[1]
fn_read = fpname + 'AsBitFingerprint.npy'
fn_read = os.path.join(directory, fn_read)
fn_save = fpname + 'AsBitFingerprint_embedding.npy'
fn_save = os.path.join(directory, fn_save)
fps_nparray = np.load(fn_read)
fps_nparray = np.asarray(fps_nparray, dtype=float)

t0 = time()
fps_tsne = bhtsne.run_bh_tsne(fps_nparray, perplexity=perplexity, theta=theta,
                              initial_dims=fps_nparray.shape[1], use_pca=True,
                              verbose=1, max_iter=1000)
t1 = time()

np.save(fn_save, fps_tsne)
print('bhtSNE embedding of the {0}AsBitFingerprints (time {1:.2f}s)'.format(fpname, t1-t0))


