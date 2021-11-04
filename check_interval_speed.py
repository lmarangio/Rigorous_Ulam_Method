from __future__ import division
from time import time


I = RIF(1/3)
two = RIF('0.5')

t = time()

for i in range(10000):
	I = I + two

print "Time: ", time() - t
