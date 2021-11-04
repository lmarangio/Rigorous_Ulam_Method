from partition import equispaced
from matrix_io import *
from ulam import *
from partition import equispaced
from generic_assembler import *
from noise_columnwise_assembler import *
import numpy as np
import os
import joblib

from bzmodel import *
from noise_computation import *


K = 2**12
prec = 128

D = BZModel(prec)
part = UlamL1(equispaced(K))

img = None

for i in range(len(part)):
    I = D.field(part.partition[i], part.partition[i+1])
    h = D.field(part.partition[i+1]) - D.field(part.partition[i])
    x = D.field(part.partition[i])

    #incratio = (D.f(x+h) - D.f(x))/h
    #deriv = D.f_prime(I)
    if False and not incratio in deriv:
        print 'Error of derivative!'
        print 'I        = [',I.lower(),',',I.upper(),']'
        print 'incratio = [',incratio.lower(),',',incratio.upper(),']'
        print 'deriv    = [',deriv.lower(),',',deriv.upper(),']'
        raise ValueError, 'morte'

    incratio = (D.f_prime(x+h) - D.f_prime(x))/h
    deriv = D.f_second(I)
        
    if not incratio in deriv:
        print 'I        = [',I.lower(),',',I.upper(),']'
        print 'incratio = [',incratio.lower(),',',incratio.upper(),']'
        print 'deriv    = [',deriv.lower(),',',deriv.upper(),']'
        print 'Error of second derivative!'
        raise ValueError, 'morte2'

    res = D.f(I)
    img = img.union(res) if img else res

if img.lower() <= 0.01:
    print 'Lower part close to the border!'
if img.upper() >= 0.99:
    print 'Upper part close to the border!'
print 'Image is [',img.lower(),',',img.upper(),']'
    
