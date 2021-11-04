"""
Makes some tests with fixed parameters to check performance of the assemblers
"""

from dynamic import *
from ulam import *
from hat_evaluation import *
from generic_assembler import *
from partition import *

import time

def timing(f, name=None):
	if name == None:
		name = f.func_name
	time1 = time.time()
	ret = f()
	time2 = time.time()
	print '%s function took %0.3f ms' % (name, (time2-time1)*1000.0)

p = equispaced(1024)
basis = UlamL1(p)

D = PerturbedFourxDynamic(c=0.3)
timing(lambda: assemble(D, basis, 1e-12), "PerturbedFourx, 0.3, 1024, UlamL1")

D = LanfordDynamic()
timing(lambda: assemble(D, basis, 1e-12), "Lanford, 1024, UlamL1")

D = MannevillePomeauDynamic()
timing(lambda: assemble(D, basis, 1e-12), "Manneville-Pomeau, 1024, UlamL1")

basis = HatFunctionsWithEvaluationsLinf(p)

D = PerturbedFourxDynamic(c=0.3)
timing(lambda: assemble(D, basis, 1e-12), "PerturbedFourx, 0.3, 1024, hat")

D = LanfordDynamic()
timing(lambda: assemble(D, basis, 1e-12), "Lanford, 1024, hat")

D = MannevillePomeauDynamic()
timing(lambda: assemble(D, basis, 1e-12), "Manneville-Pomeau, 1024, hat")

