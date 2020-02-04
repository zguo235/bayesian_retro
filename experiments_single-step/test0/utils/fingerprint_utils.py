import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix
from rdkit import Chem
from rdkit.Chem import AllChem


class SparseFingerprintCsrMatrix(object):

    def __init__(self, smis=None, mols=None, fp_matrix=None,
                 col_length=2**16, fp_radius=2, **fp_args):
        self.smis = smis
        self.mols = mols
        self.fp_matrix = fp_matrix
        self.col_length = col_length
        self.fp_radius = fp_radius
        self.fp_args = fp_args

    def tocsr(self):
        if self.fp_matrix is None:
            self.calc_fp_matrix()
        indices = list()
        data = list()
        indptr = [0]
        for i, fp in enumerate(self.fp_matrix):
            fp_nonzero = fp.GetNonzeroElements()
            fp_new = defaultdict(lambda: 0)
            for key, value in fp_nonzero.items():
                fp_new[key % self.col_length] += value
            indices.extend(fp_new.keys())
            data.extend(fp_new.values())
            indptr.append(len(fp_new))
        indptr = np.cumsum(indptr)
        M = len(self.fp_matrix)
        N = self.col_length
        return csr_matrix((data, indices, indptr), shape=(M, N), dtype=np.float32)

    def calc_fp_matrix(self):
        if self.mols is None:
            self.parse_mol_matrix()
        self.fp_matrix = [AllChem.GetMorganFingerprint(mol, radius=self.fp_radius,
                                                       **self.fp_args) for mol in self.mols]

    def parse_mol_matrix(self):
        self.mols = [Chem.MolFromSmiles(smi) for smi in self.smis]


class BitFingerprintCsrMatrix(object):

    def __init__(self, smis=None, mols=None, fp_matrix=None):
        self.smis = smis
        self.mols = mols
        self.fp_matrix = fp_matrix

    def tocsr(self):
        if self.fp_matrix is None:
            self.calc_fp_matrix()
        indices = list()
        indptr = [0]
        for i, fp in enumerate(self.fp_matrix):
            indices.extend(fp.GetOnBits())
            indptr.append(fp.GetNumOnBits())
        data = [1] * len(indices)
        indptr = np.cumsum(indptr)
        M = len(self.fp_matrix)
        N = self.fp_matrix[0].GetNumBits()
        return csr_matrix((data, indices, indptr), shape=(M, N), dtype=np.float32)

    def calc_fp_matrix(self):
        if self.mols is None:
            self.parse_mol_matrix()
        self.fp_matrix = [AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024,
                                                                useFeatures=False) for mol in self.mols]

    def parse_mol_matrix(self):
        self.mols = [Chem.MolFromSmiles(smi) for smi in self.smis]


def csc_drop_zerocols(matrix):
    nonzero_cols = np.diff(matrix.indptr) != 0
    new_indptr = matrix.indptr[np.append(True, nonzero_cols)]
    new_shape = (matrix.shape[0], np.count_nonzero(nonzero_cols))
    return csc_matrix((matrix.data, matrix.indices, new_indptr), shape=new_shape)


def coo_drop_zerocols(matrix):
    nz_cols, new_cols = np.unique(matrix.col, return_inverse=True)
    new_shape = (matrix.shape[0], len(nz_cols))
    return coo_matrix((matrix.data, (matrix.row, new_cols)), shape=new_shape)
