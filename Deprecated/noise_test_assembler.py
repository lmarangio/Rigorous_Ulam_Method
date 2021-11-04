from partition import equispaced
from matrix_io import save_matrix_market, save_binary_matrix, dump_while_assembling
from sparse import sage_sparse_to_scipy_sparse, sparse_matvec,\
     interval_norm1_error, max_nonzeros_per_row, norm1, interval_norminf_error
from ulam import *
from partition import equispaced
from generic_assembler import *
from noise_columnwise_assembler import *
import numpy as np
import os
import joblib

from bzmodel import *
from noise_computation import *


K = 2**15
prec = 256

D = BZModel(prec)

epsilon = D.field(2) ** (20-prec)
ulam_basis = UlamL1(equispaced(K))

P, absolute_diameter, nnz, norm1_error = columnwise_assemble(D, ulam_basis, epsilon,
                                                             scipy_matrix = False,
                                           verification_basis = ('tests/verif_%d' % K))

save_binary_matrix(P, 'tests/mat_%d' % K, use_double = True)

#save_scipy_to_binary_matrix(P, 'tests/mat_%d' % K, use_double = True)

Q = assemble(D, ulam_basis, epsilon)

Delta = P-Q
diff = max(abs(Delta[i,j]) for i,j in Delta.nonzero_positions(copy=False))
print 'Diff:',diff
