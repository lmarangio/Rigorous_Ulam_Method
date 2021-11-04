
from sage.rings.real_mpfr import RealField
import time

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

def show_progress(start_time, i, total):
    elapsed = time.time() - start_time
    to_completion = elapsed / i * (total-i)
    return "%.02f%%, elapsed: %s, ETA: %s" % (
                   RealField(prec=53)(i*100.0/total),
                   show_time(elapsed),
                   show_time(to_completion) )
