
from partition import equispaced
from matrix_io import save_matrix_market, save_binary_matrix, dump_while_assembling
from sparse import sage_sparse_to_scipy_sparse, sparse_matvec,\
     interval_norm1_error, max_nonzeros_per_row, norm1, interval_norminf_error
from ulam import *
from partition import equispaced
import numpy as np
import os
import joblib
import gc

import time
from matrix_io import *
from test1 import *
from noise_measure_computation import *
from noise_observable_estimation import *


def montecarlo_obs(D, obs, noise, reps = 5, eps = 0.06, steps = 100):
    RI = D.field
    xs = [RI(random()) for _ in range(reps)]
    sums = [RI(0) for _ in range(reps)]
    its = 0
    
    while True:
        new_sums = [s for s in sums]
        for _ in range(steps):
            for i in range(reps):
                xs[i] = RI(D.f(xs[i]).center()+(random()-0.5)*noise)
                #xs[i] = RI(D.f(xs[i]).center()+random()*noise)
                new_sums[i] += obs(xs[i])
            #print xs
            #print new_sums
        if its > 0:
            avgs = [s/its for s in sums] + [s/(its+steps) for s in new_sums]
            retv = RI(min(avgs).lower(), max(avgs).upper())
            print '[',noise,'] Diam:',retv.diameter(),' after',its+steps,'steps, ~',\
                (sum(avgs)/len(avgs)).center()
            if retv.diameter() < eps:
                print retv.lower(),retv.upper()
                return retv
        its += steps
        sums = new_sums
        #break

#alphas = ['0.35', '0.4', '0.45', '0.5', '0.56', '0.62', '0.68']
#lambds = ['0.4', '0.455', '0.51', '0.566', '0.62', '0.7', '0.78']
#betas =  ['90', '105', '120', '137', '150']
#noises = [0.1, 0.063, 0.04, 0.025, 0.016, 0.01, 0.0063, 0.004, 0.0025, 0.0016, 0.001]
#noises = [0.1**(RR(x)/10) for x in range(10, 41)]

#for alpha in alphas:
#  for lambd in lambds:
#    for beta in betas:

#for alpha, lambd, beta in params:
#        print alpha, lambd, beta

D = Test1(53)
obs = lambda x: D.f_prime(x).abs().log()

#noises = [0.1**(RR(x)/4) for x in range(4, 13)]
noises = [0.1, 0.01, 0.001]

lyaps = [montecarlo_obs(D, obs, n).center() for n in noises]
myplot = plot([])
for i in range(len(lyaps)-1):
    myplot += line([(noises[i], lyaps[i]), (noises[i+1], lyaps[i+1])],
                   color="#f00")
#myplot.show(scale = "semilogx")        
myplot.save_image('test1_plot.png', scale = "semilogx")

#f = open('nagumosato_montecarlo.txt', 'a')
#f.write("%s %s %s : %s\n" % (alpha, lambd, beta, ' '.join(lyaps)))
#f.close()
