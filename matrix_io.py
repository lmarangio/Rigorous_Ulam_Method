"""
Functions to save a matrix as matrixmarket or binary
"""
import scipy.io
import struct
import time
import shutil
import os
import joblib
import numpy as np
from sage.all import VectorSpace
from sage.rings.real_mpfr import RealField
from sage.rings.real_mpfi import is_RealIntervalFieldElement
from sparse import sage_sparse_to_scipy_sparse

#__all__ = ["save_matrix_market", "save_binary_matrix", "save_scipy_to_binary_matrix"]

def save_matrix_market(P, path):
    PP = sage_sparse_to_scipy_sparse(P)
    scipy.io.mmwrite(path, PP)


class MatrixBinaryWriter:
    def __init__(self, path, height, width, use_double=True):
        self.path = path
        self.height = height
        self.width = width
        self.nnz = 0
        self.float_fmt = 'd' if use_double else 'f'
        self.rows = file(self.path+'.rows', 'w')
        self.cols = file(self.path+'.cols', 'w')
        self.vals = file(self.path+'.vals', 'w')

    def write(self, i, j, val):
        self.nnz += 1
        self.rows.write(struct.pack('i',i))
        self.cols.write(struct.pack('i',j))
        self.vals.write(struct.pack(self.float_fmt,val))

    def close(self):
        main = file(self.path, 'w')
        main.write("%d %d %d %s\n" % (self.height, self.width, self.nnz, self.float_fmt))
        main.close()
        self.rows.close()
        self.cols.close()
        self.vals.close()

def show_time(t):
    secs = t % 60
    retv = "%.02fs" % secs
    mins = int(t / 60)
    if mins > 0:
        retv = ("%dm "%(mins%60)) + retv
    hours = int(t / 3600)
    if hours > 0:
        retv = ("%dh "%hours) + retv
    return retv

def save_binary_matrix(P, path, use_double = True):
    height, width = P.dimensions()
    mbw = MatrixBinaryWriter(path, height, width, use_double)
    for i,j in P.nonzero_positions(copy=False):
        v = P[i,j]
        if is_RealIntervalFieldElement(v):
            v = v.center()
        mbw.write(i,j,v)
    mbw.close()

def save_scipy_to_binary_matrix(P, path, use_double = True):
    height, width = P.shape
    mbw = MatrixBinaryWriter(path, height, width, use_double)
    ijs = P.nonzero()
    for q in range(len(ijs[0])):
        i,j = ijs[0][q], ijs[1][q]
        mbw.write(i,j,P[i,j])
    mbw.close()

def mmap_binary_vector(size, file, fmt = 'd'):
    dtype = 'float64' if fmt == 'd' else 'float32'
    return np.memmap(file, dtype=dtype, mode='r', shape=(size))

def read_binary_vector(field, size, file, fmt = 'd'):
    vec = VectorSpace(field, size)(0)
    real_struct = struct.Struct(fmt)
    f = open(file, 'rb')
    ptsize = real_struct.size
    for q in range(size):
        vec[q] = real_struct.unpack(f.read(ptsize))[0]
    f.close()
    return vec

def save_binary_vector(vec, path, fmt = 'd'):
    f = file(path, 'w')
    real_struct = struct.Struct(fmt)
    for i in range(len(vec)):
        f.write(real_struct.pack(vec[i]))
    f.close()

def mmap_binary_interval_vector(size, file, fmt = 'd'):
    dtype = 'float64' if fmt == 'd' else 'float32'
    return np.memmap(file, dtype=dtype, mode='r', shape=(size, 2))

def read_binary_interval_vector(field, size, file, fmt = 'd'):
    vec = VectorSpace(field, size)(0)
    real_struct = struct.Struct(fmt)
    f = open(file, 'rb')
    ptsize = real_struct.size
    for q in range(size):
        l = real_struct.unpack(f.read(ptsize))[0]
        u = real_struct.unpack(f.read(ptsize))[0]
        vec[q] = field(l, u)
    f.close()
    return vec

def save_binary_interval_vector(vec, path, fmt = 'd'):
    f = file(path, 'w')
    real_struct = struct.Struct(fmt)
    for i in range(len(vec)):
        f.write(real_struct.pack(vec[i].lower()))
        f.write(real_struct.pack(vec[i].upper()))
    f.close()
            
def join_assembly_parts(target, paths):
    trows = open(target + '.rows', 'wb')
    tcols = open(target + '.cols', 'wb')
    tvals = open(target + '.vals', 'wb')

    tw = None
    th = 0
    tn = 0
    tt = None
    tcol_diam = None
    tmax_nnz = None
    tabsolute_diameter = None

    for i in range(len(paths)):
        p = paths[i]
        shutil.copyfileobj(open(p + '.rows', 'rb'), trows)
        shutil.copyfileobj(open(p + '.cols', 'rb'), tcols)
        shutil.copyfileobj(open(p + '.vals', 'rb'), tvals)
        pw,ph,pn,pt = open(p, 'r').readline().split()
        pw = int(pw)
        ph = int(ph)
        pn = int(pn)
        tn += pn
        #pcol_diam, pmax_nnz, pabsolute_diameter = joblib.load(p+'.info')
        pmax_nnz, pabsolute_diameter = joblib.load(p+'.info')

        if i == 0:
            tw = pw
            th = ph
            tt = pt
            tcol_diam = VectorSpace(RealField(rnd='RNDU', prec=53), pw)(0)
            tmax_nnz = pmax_nnz
            tabsolute_diameter = pabsolute_diameter
        else:
            if tw != pw or tt != pt or th != ph:
                raise ValueError,'incompatible matrices!'
            tmax_nnz = max(tmax_nnz, pmax_nnz)
            tabsolute_diameter = max(tabsolute_diameter, pabsolute_diameter)

        partial_col_diam = mmap_binary_vector(pw, p+'.partial_col_diam', pt)
        for q in range(pw):
            tcol_diam[q] += partial_col_diam[q]
        
        print 'added file ',p

    trows.close()
    tcols.close()
    tvals.close()
    tmain = open(target, 'w')
    tmain.write("%d %d %d %s\n" % (tw,th,tn,tt))
    tmain.close()

    tnorm1_error = max(tcol_diam)
    print "absolute diameter", tabsolute_diameter
    print "max non-zeros", tmax_nnz
    print "partial norm1 error", tnorm1_error
    return tabsolute_diameter, tmax_nnz, tnorm1_error


def dump_while_assembling(path, dynamic, basis, epsilon,
                          begin_i = 0, end_i = None,
                          use_double = True, output_rate=1024):
    print "Epsilon ", epsilon

    start_time = time.time()
    absolute_diameter = 0
    max_nnz = 0

    K   = len(basis)
    mbw = MatrixBinaryWriter(path, K, K, use_double)

    i = begin_i
    if not end_i:
        end_i = K

    if output_rate < (end_i - begin_i) / 64:
        output_rate = (end_i - begin_i) / 64

    #col_diam = VectorSpace(RealField(rnd='RNDU', prec=dynamic.field.prec()), K)(0)
    col_diam = VectorSpace(RealField(rnd='RNDU', prec=53), K)(0)

    for dual_element in basis.dual_composed_with_dynamic(dynamic, epsilon, begin_i, end_i):
        nnz = 0
        for j,x in basis.project_dual_element(dual_element):
            mbw.write(i, j, x.center())
            nnz += 1
            xdiam = x.absolute_diameter()
            absolute_diameter = max(absolute_diameter, xdiam)
            col_diam[j] += xdiam

        max_nnz = max(max_nnz, nnz)
        i += 1
        if (output_rate is not 0) and (i%output_rate==0):
            elapsed = time.time() - start_time
            to_completion = elapsed / (i-begin_i) * (end_i-i)
            print ("%d-%d: %d (%.02f%%, elapsed: %s, ETA: %s) " % (
                   begin_i, end_i, i-begin_i,
                   RealField(prec=53)((i-begin_i)*100.0/(end_i-begin_i)),
                   show_time(elapsed),
                   show_time(to_completion)
            ) )
    mbw.close()

    #Rup = RealField(rnd='RNDU')
    col_diam_dump = file(path+'.partial_col_diam', 'w')
    fmt = 'd' if use_double else 'f'
    for i in range(len(col_diam)):
        col_diam_dump.write(struct.pack(fmt, col_diam[i]))
    col_diam_dump.close()

    #joblib.dump([col_diam, max_nnz, absolute_diameter], path+'.info')
    joblib.dump([max_nnz, absolute_diameter], path+'.info')
    
    norm1_error = max(col_diam)
    print "absolute diameter", absolute_diameter
    print "max non-zeros", max_nnz
    print "partial norm1 error", norm1_error
    return absolute_diameter, max_nnz, norm1_error
