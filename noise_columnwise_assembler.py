
from sage.all import *
from matrix_io import show_time
from scipy.sparse import *
from joblib import Parallel, delayed
import struct
from functools import partial

import numpy as np
import time
from show_progress import *
load('binsearch.spyx')

def compute_column(dynamic, basis, epsilon, verification_basis, i1, i2):
    res = []

    for i in range(i1, i2):
        RI = dynamic.field
        K = len(basis)

        a = RI(basis.partition[i])
        b = RI(basis.partition[i+1])
        factor = ~(b-a) # 1 / delta
    
        contributes = [] if verification_basis else None
        column = VectorSpace(RI, K)(0)

        for br_num in range(dynamic.nbranches):
            br_interval = dynamic.branch_intervals[br_num]
            if br_interval[1].upper() <= a.lower() or \
               br_interval[0].lower() >= b.upper():
                continue

            br_a = max(a, br_interval[0])
            br_b = min(b, br_interval[1])
            
            br_Ta = dynamic.f(a)
            br_Tb = dynamic.f(b)
            is_decreasing = br_Ta.lower() > br_Tb.upper()

            br_range = br_Ta.union(br_Tb)
            
	    jmin = binsearch(br_range.lower(), basis.partition)
	    jmax = binsearch(br_range.upper(), basis.partition)

            br_contributes = [] if verification_basis else None
            prev_preimage = br_b if is_decreasing else br_a
            for j in range(jmin, jmax+1):
                if basis.partition[j+1] <= br_range.lower():
                    next_preimage = br_b if is_decreasing else br_a
                elif basis.partition[j+1] >= br_range.upper():
                    next_preimage = br_a if is_decreasing else br_b
                else:
                    next_preimage = dynamic.preimage(basis.partition[j+1], br_num, epsilon)

                    # entry to add to the Ulam matrix
                entry = abs(next_preimage-prev_preimage) * factor
                if entry.lower() > 1:
                    print a,b
                    print br_a,br_b
                    print br_Ta,br_Tb
                    print 'decr:',is_decreasing, 'br:',br_num
                    print '[',br_Ta.lower(),',',br_Ta.upper(),']'
                    print '[',br_Tb.lower(),',',br_Tb.upper(),']'
                    print j,' in ',jmin, jmax
                    print basis.partition[j],' -> ', prev_preimage
                    print basis.partition[j+1],' -> ', next_preimage
                    raise 'uatsefack'

                column[j] += entry

                if verification_basis:
                    br_contributes.append( (j, entry) )

                prev_preimage = next_preimage

            # add the contributes of this branch, in the correct order
            if verification_basis:
                if is_decreasing:
                    br_contributes.reverse()
                contributes += br_contributes

        res.append( (column, contributes, i) )
    return res

# Very slow using parallel, unluckily
def columnwise_assemble_try(dynamic, basis, epsilon,
                        verification_basis = None,
                        scipy_matrix = True,
                        float_fmt = 'd'):
    RI = dynamic.field
    K = len(basis)

    # for checking purpose...
    if scipy_matrix:
	absolute_diameter = 0
        norm1_error = 0
        row_nnz = VectorSpace(RR, K)(0)
        P = lil_matrix((K,K))
        #P = matrix(RR, K, K, sparse=True)
    else:
        P = matrix(dynamic.field, K, K, sparse=True)

    if verification_basis:
        verif_num_vecs_written = 0
        verif_num_vals_written = 0
        verif_idx = file(verification_basis+'.idx','w')
        verif_rows = file(verification_basis+'.rows','w')
        verif_vals = file(verification_basis+'.vals','w')

    start_time = time.time()

    my_compute = partial(compute_column, dynamic, basis, epsilon, verification_basis)
    
    #for i in range(K):
    for stride_i in range(0, K, 512):
      ress = Parallel(n_jobs=2, max_nbytes='512M', backend='threading')(
                          delayed(compute_column)(dynamic, basis,
                                                  epsilon, verification_basis, i, i+64)
                                for i in range(stride_i, stride_i+512, 64))

      for res in ress:
       for column, contributes, i in res:
           #for i in range(stride_i, stride_i+128):
           #column, contributes = res[i - stride_i]
        

	if scipy_matrix:
            thiscol_norm1_error = 0
        for j in column.nonzero_positions():
            x = column[j]
	    if scipy_matrix:
		P[j,i] = RDF(x.center())
                thiscol_norm1_error += x.absolute_diameter()
		absolute_diameter = max(absolute_diameter, x.absolute_diameter())
                row_nnz[j] += 1
            else:
		P[j,i] = x
	if scipy_matrix:
            norm1_error = max(norm1_error, thiscol_norm1_error)
                
        if verification_basis:
            for r in range(1, len(contributes)):
                s1 = RI(2) * sum(contributes[q][1] for q in range(r))
                s2 = RI(2) * sum(contributes[q][1] for q in range(r, len(contributes)))

                vec = VectorSpace(dynamic.field, K)(0)
                for q in range(r):
                    vec[ contributes[q][0] ] += -s2 * contributes[q][1]
                for q in range(r, len(contributes)):
                    vec[ contributes[q][0] ] += s1 * contributes[q][1]
                nz = vec.nonzero_positions()
                verif_idx.write(struct.pack('i', verif_num_vals_written))
                verif_num_vecs_written += 1
                for j in nz:
                    verif_rows.write(struct.pack('i', j))
                    verif_vals.write(struct.pack(float_fmt, RR(vec[j].center())))
                    verif_num_vals_written += 1
                    #verif.write("%d\t%s\n" % (j, str(RR(vec[j].center()))))
            
      #if (i+1) % 256 == 0:
      #print show_progress(start_time, i+1, K)
      print show_progress(start_time, stride_i+512, K)

    if verification_basis:
        verif_idx.write(struct.pack('i', verif_num_vals_written))
        verif_idx.close()
        verif_rows.close()
        verif_vals.close()
        verif = file(verification_basis, 'w')
        verif.write("%d %d %s\n" % (verif_num_vecs_written,
                                    verif_num_vals_written, float_fmt))
        verif.close()

    if scipy_matrix:
        nnz = max(row_nnz)
        return P, absolute_diameter, nnz, norm1_error
    else:
        return P


def columnwise_assemble(dynamic, basis, epsilon,
                        verification_basis = None,
                        scipy_matrix = True,
                        float_fmt = 'd'):
    RI = dynamic.field
    K = len(basis)

    # for checking purpose...
    if scipy_matrix:
	absolute_diameter = 0
        norm1_error = 0
        row_nnz = VectorSpace(RR, K)(0)
        P = lil_matrix((K,K))
        #P = matrix(RR, K, K, sparse=True)
    else:
        P = matrix(dynamic.field, K, K, sparse=True)

    if verification_basis:
        verif_num_vecs_written = 0
        verif_num_vals_written = 0
        verif_idx = file(verification_basis+'.idx','w')
        verif_rows = file(verification_basis+'.rows','w')
        verif_vals = file(verification_basis+'.vals','w')
        verif_refs = file(verification_basis+'.refs','w')

    start_time = time.time()

    output_rate = K/64

    for i in range(K):
        a = RI(basis.partition[i])
        b = RI(basis.partition[i+1])
        factor = ~(b-a) # 1 / delta

        contributes = []
        column = VectorSpace(dynamic.field, K)(0)

        for br_num in range(dynamic.nbranches):
            br_interval = dynamic.branch_intervals[br_num]
            if br_interval[1].upper() <= a.lower() or \
               br_interval[0].lower() >= b.upper():
                continue

            br_a = max(a, br_interval[0])
            br_b = min(b, br_interval[1])

            if 'f_branch' in dir(dynamic): # use f_branch if available
                br_Ta = dynamic.f_branch(a, br_num)
                br_Tb = dynamic.f_branch(b, br_num)
            else:
                br_Ta = dynamic.f(a)
                br_Tb = dynamic.f(b)
            is_decreasing = br_Ta.lower() > br_Tb.upper()

            br_range = br_Ta.union(br_Tb)
            
	    jmin = binsearch(br_range.lower(), basis.partition)
	    jmax = binsearch(br_range.upper(), basis.partition)

            br_contributes = []
            prev_preimage = br_b if is_decreasing else br_a
            for j in range(jmin, jmax+1):
                if basis.partition[j+1] <= br_range.lower():
                    next_preimage = br_b if is_decreasing else br_a
                elif basis.partition[j+1] >= br_range.upper():
                    next_preimage = br_a if is_decreasing else br_b
                else:
                    next_preimage = dynamic.preimage(basis.partition[j+1], br_num, epsilon)

                # entry to add to the Ulam matrix
                entry = abs(next_preimage-prev_preimage) * factor
                if entry.lower() > 1:
                    print a,b
                    print br_a,br_b
                    print br_Ta,br_Tb
                    print 'decr:',is_decreasing, 'br:',br_num
                    print '[',br_Ta.lower(),',',br_Ta.upper(),']'
                    print '[',br_Tb.lower(),',',br_Tb.upper(),']'
                    print j,' in ',jmin, jmax
                    print basis.partition[j],' -> ', prev_preimage
                    print basis.partition[j+1],' -> ', next_preimage
                    raise 'uatsefack'

                #P[j,i] += RDF(entry.center())
                column[j] += entry

                br_contributes.append( (j, entry) )
                prev_preimage = next_preimage

            # add the contributes of this branch, in the correct order
            if verification_basis:
                if is_decreasing:
                    br_contributes.reverse()
                contributes += br_contributes


	if scipy_matrix:
            thiscol_norm1_error = 0
        for j in column.nonzero_positions():
            x = column[j]
	    if scipy_matrix:
		P[j,i] = RDF(x.center())
                thiscol_norm1_error += x.absolute_diameter()
		absolute_diameter = max(absolute_diameter, x.absolute_diameter())
                row_nnz[j] += 1
            else:
		P[j,i] = x
	if scipy_matrix:
            norm1_error = max(norm1_error, thiscol_norm1_error)
                
        if verification_basis:

            for r in range(1, len(contributes)):
                s1 = RI(2) * sum(contributes[q][1] for q in range(r))
                s2 = RI(2) * sum(contributes[q][1] for q in range(r, len(contributes)))

                vec = VectorSpace(dynamic.field, K)(0)
                for q in range(r):
                    vec[ contributes[q][0] ] += -s2 * contributes[q][1]
                for q in range(r, len(contributes)):
                    vec[ contributes[q][0] ] += s1 * contributes[q][1]
                nz = vec.nonzero_positions()
                verif_idx.write(struct.pack('i', verif_num_vals_written))
                verif_refs.write(struct.pack('i', i)) #from what element are we originating?
                verif_num_vecs_written += 1
                for j in nz:
                    verif_rows.write(struct.pack('i', j))
                    verif_vals.write(struct.pack(float_fmt, RR(vec[j].center())))
                    verif_num_vals_written += 1
                    #verif.write("%d\t%s\n" % (j, str(RR(vec[j].center()))))
            
        if (i+1) % output_rate == 0:
            print show_progress(start_time, i+1, K)

    if verification_basis:
        verif_idx.write(struct.pack('i', verif_num_vals_written))
        verif_idx.close()
        verif_rows.close()
        verif_vals.close()
        verif = file(verification_basis, 'w')
        verif.write("%d %d %s\n" % (verif_num_vecs_written,
                                    verif_num_vals_written, float_fmt))
        verif.close()

    if scipy_matrix:
        nnz = max(row_nnz)
        return P, absolute_diameter, nnz, norm1_error
    else:
        return P
