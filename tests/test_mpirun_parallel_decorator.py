import numpy as np
import sys
sys.path.append('/mnt/d/Python/minifympi')
from minifympi.core.base import MinifyMPI, parallel
from numba import jit

n_procs = 4
parallel.mmp = MinifyMPI()

@parallel(n_procs)
def test(a:'S', b, c:'Sv')->'g':
    # c = np.arange(np.random.randint(1, 4, 1))
    return a+b, b, c

a = np.arange(n_procs)
b = np.arange(n_procs) * 5
c = np.arange(10)

# print(a, b)
res = test(a, b, c)
print(res)





