import sys
sys.path.append("/mnt/d/Python/minifympi/")

from minifympi.core import MinifyMPI
from mpi4py import MPI
import numpy as np
import time


n_procs = 4
n_tasks = 8
mmp = MinifyMPI(n_procs, n_tasks, 'mpirun')
mmp.start_comm({'mmp': mmp})

@mmp.parallel
def test(a:'S', b:'s', c=None, d:'S'=None, *args, **kwargs)->'G,g,g':
    print(a[0, 0, 0, :3])
    print(mmp.rank, mmp.comm.rank)

    return a, b, d

a = np.random.randint(0, 1000, (n_tasks, 1024, 1024, 512//8))
d = np.random.randint(0, 1000, (n_tasks, 1024, 1024, 512//8))
b = list(range(n_procs))
c = np.arange(n_procs*2).reshape(n_procs, -1)

t0 = time.time()
mmp.Scatter['a', 'd'] = a, d
mmp.scatter['b'] = b
mmp.Bcast['c'] = c

mmp.test['res1', 'res2', 'res3'](mmp['a'], mmp['b'], mmp['c'], mmp['d'], 5, 10, 11)

res1, res3 = mmp.Gather['res1', 'res3']
res2 = mmp.gather['res2']
mmp.close_comm()

t1 = time.time()
print('cost', t1-t0, 's')