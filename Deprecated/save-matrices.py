"""
Functions to write on disk a bunch of assembled matrices in Matrix Market format
"""

from dynamic import *
from assembler import *
from scipy.io import mmwrite
from sparse import sage_sparse_to_scipy_sparse

prec = 53
n_jobs = 4
epsilon=1e-10

D = PerturbedFourxDynamic(prec=prec, c = 0.4)

filenameFormat = "P%d.mtx"

def assemble_and_write_to_disk(k, epsilon, prec, n_jobs):
	"""
	Assemble a single 2^k x 2^k matrix and writes it to a file.
	"""
	P = assemble(D, 2**k, epsilon=epsilon, prec=prec, n_jobs=n_jobs, check_partition=False)
	P = P.apply_map(lambda x: RR(x.center()))
	P = sage_sparse_to_scipy_sparse(P)
	filename = filenameFormat % 2**k
	mmwrite(filename, P)

for k in range(17,20):
	assemble_and_write_to_disk(k, epsilon, prec, n_jobs)
