import sys
import numpy as np
sys.path.append("/mnt/d/Python/minifympi/")
from minifympi.wrappers import parallel
from mpi4py import MPI
import time
from numba import njit

n_procs = 4
n_tasks = 8
@parallel(n_procs, n_tasks)
def test(a:'S', b:'s', c=None, d:'S'=None, *args, **kwargs)->'G,g,G':
    print(a.shape)
    return a, b, d

# a = np.arange(n_tasks*2).reshape(n_tasks, -1)
a = np.random.randint(0, 1000, (n_tasks, 512, 512, 512//128))
d = np.random.randint(0, 1000, (n_tasks, 512, 512, 512//128))
b = list(range(n_procs))
c = np.arange(n_procs*2).reshape(n_procs, -1)
t0 = time.time()
returns = test(a, b, c, d, 5, 10, 11)


t1 = time.time()
print('cost', t1-t0, 's')
print(returns[0][0, 0, 0, :3])
for i, item in enumerate(returns):
    if isinstance(item, list):
        print(f'item{i}', type(item), len(item))
    else:
        print(f'item{i}', type(item), item.shape)


