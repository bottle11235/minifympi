import numpy as np
import sys
sys.path.append('/mnt/d/Python/minifympi')
from minifympi.core.notebook import parallel
from numba import jit

n_procs = 4

@parallel(n_procs)
@jit
def test(a:'S', b, c:'Sv')->'G':
    # c = np.arange(np.random.randint(1, 4, 1))
    return a+b

a = np.arange(n_procs)
b = np.arange(n_procs*2) * 5
c = np.arange(10)


# print(a, b)
res = test(a, b, c)
print(res)





