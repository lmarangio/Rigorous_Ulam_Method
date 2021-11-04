"""
Simple function to profile the assembler.

Requires runsnake.
"""

from assembler import assemble
from dynamic import *

prec = 53
epsilon = 1e-10

D = PerturbedFourxDynamic(prec=prec, c = 0.4)
runsnake('assemble(D, 2**16, epsilon=epsilon, prec=prec, n_jobs=1, do_check_partition=False)')
