
from partition import equispaced
from matrix_io import save_matrix_market, save_binary_matrix, dump_while_assembling,\
    join_assembly_parts
from sparse import sage_sparse_to_scipy_sparse, sparse_matvec,\
     interval_norm1_error, max_nonzeros_per_row, norm1, interval_norminf_error
from ulam import *
from joblib import Parallel, delayed
from partition import equispaced
import numpy as np
import os
import joblib

# computes the step error in an application of the matrix P (no noise)
def compute_basic_step_error(K, abs_diam, max_nnz, norm1_error):
    Rup = RealField(rnd='RNDU')
    Rdown = RealField(rnd='RNDD')

    machineEpsilon = Rup(np.finfo(float).eps)

    gammak = machineEpsilon*max_nnz / (1-machineEpsilon*max_nnz) + machineEpsilon
    steperror = Rup(norm1_error) + Rup(gammak)

    return steperror


# creates the matrix, avoids recreating it if it is already saved somewhere
def create_matrix(D, K, prec):

    # create dir if necessary
    if not os.path.exists(D.name):
        os.makedirs(D.name)

    filename = D.name+'/mat_%d_%d' % (K, prec)
    data_filename = filename + '_data'

    if os.access(filename, os.R_OK):
        basic_step_error, abs_diam, max_nnz, norm1_error = joblib.load(data_filename)
        return filename, basic_step_error
    else:
        print '* assembling matrix'
        epsilon = D.field(2) ** (25-prec)

        #P = assemble(D, K, epsilon = epsilon, prec = prec, n_jobs=1)

        #print '* saving the matrix as binary'
        #save_binary_matrix(P, filename, True)

        abs_diam, max_nnz, norm1_error = \
            dump_while_assembling(filename, D, UlamL1(equispaced(K)), epsilon)

        basic_step_error = compute_basic_step_error(K, abs_diam,
                                                    max_nnz, norm1_error)
        joblib.dump([basic_step_error, abs_diam, max_nnz, norm1_error], data_filename)

        return filename, basic_step_error


def create_matrix_from_parts(D, K, prec, parts):

    # create dir if necessary
    if not os.path.exists(D.name):
        os.makedirs(D.name)

    filename = D.name+'/mat_%d_%d' % (K, prec)
    data_filename = filename + '_data'

    print '* joining matrix parts'

    abs_diam, max_nnz, norm1_error = \
        join_assembly_parts(filename, parts)

    basic_step_error = compute_basic_step_error(K, abs_diam,
                                                max_nnz, norm1_error)
    joblib.dump([basic_step_error, abs_diam, max_nnz, norm1_error], data_filename)

    return filename, basic_step_error


def create_matrix_part(D, K, prec, block_index = 0, num_blocks = 1):
    begin_i = K/1024 * ((1024*block_index) / num_blocks)
    end_i = K/1024 * ((1024*(1+block_index)) / num_blocks)

    # create dir if necessary
    if not os.path.exists(D.name):
        os.makedirs(D.name)

    filename = D.name+'/noise_mat_%d_%d_%dto%d' % (K, prec, begin_i, end_i)
    epsilon = D.field(2) ** (25-prec)

    dump_while_assembling(filename, D, UlamLazyEquispaced(K, D.field), epsilon, begin_i, end_i)

# Duh!! Very slow using 'Parallel'. Call create_matrix_part in
# different sage processes instead, to create matrix by pieces.
def create_matrix_parts(D, K, prec, block_index_begin = 0,
                        block_index_end = 0, num_blocks = 1):
    Parallel(n_jobs=(block_index_end-block_index_begin+1),
             max_nbytes='512M', backend='threading')(
                 delayed(create_matrix_part)(D, K, prec, i, num_blocks)
                 for i in range(block_index_begin, block_index_end))
