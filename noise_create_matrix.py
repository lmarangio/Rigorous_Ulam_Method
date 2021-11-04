
from partition import equispaced
from matrix_io import save_matrix_market, save_binary_matrix, dump_while_assembling
from sparse import sage_sparse_to_scipy_sparse, sparse_matvec,\
     interval_norm1_error, max_nonzeros_per_row, norm1, interval_norminf_error
from ulam import *
from joblib import Parallel, delayed
from partition import equispaced
import numpy as np
import os, sys
import joblib
from noise_matrix_creation import *
from noise_contraction_proof import *
from bzmodel import *

K = 2**27
prec = 256
D = BZModel(prec)

# nothing intersting in this file.
# only useful if you are creating a really HUGE matrix (>=16M)
# and want to do so creating a piece at a time.

#create_matrix(D, K, prec)
#create_matrix_part(D, K, prec, 1, 8)
#sys.exit()
#create_contraction_matrix(D, 2**19, prec)
#print prove_contraction(D, 2**19, 184, 65)
#print prove_contraction(D, 2**19, 140, 65)
#print prove_contraction(D, 2**19, 108, 65)
#create_contraction_matrix(D, 2**21, prec)

#parts = ['bzmodel/noise_mat_33554432_256_0to2785280',
#         'bzmodel/noise_mat_33554432_256_2785280to5570560',
#         'bzmodel/noise_mat_33554432_256_5570560to11173888',
#         'bzmodel/noise_mat_33554432_256_11173888to16777216',
#         'bzmodel/noise_mat_33554432_256_16777216to22347776',
#         'bzmodel/noise_mat_33554432_256_22347776to27951104',
#         'bzmodel/noise_mat_33554432_256_27951104to33554432']

#parts = [
#    'bzmodel/noise_mat_67108864_256_0to9568256',
#    'bzmodel/noise_mat_67108864_256_9568256to19136512',
#    'bzmodel/noise_mat_67108864_256_19136512to28704768',
#    'bzmodel/noise_mat_67108864_256_28704768to38338560',
#    'bzmodel/noise_mat_67108864_256_38338560to47906816',
#    'bzmodel/noise_mat_67108864_256_47906816to57475072',
#    'bzmodel/noise_mat_67108864_256_57475072to67108864'
#]

parts = [
    'bzmodel/noise_mat_134217728_256_0to16777216',
    'bzmodel/noise_mat_134217728_256_16777216to33554432',
    'bzmodel/noise_mat_134217728_256_33554432to67108864',
    'bzmodel/noise_mat_134217728_256_67108864to100663296',
    'bzmodel/noise_mat_134217728_256_100663296to134217728'
]
create_matrix_from_parts(D, K, prec, parts)
