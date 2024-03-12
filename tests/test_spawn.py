#!/usr/bin/env python
from mpi4py import MPI
import numpy as np
import sys
import time

comm = MPI.COMM_SELF.Spawn(sys.executable,
                           args=['sub.py'],
                           maxprocs=4)

n_tasks = 8
n_procs = 4
a_send = np.random.randint(0, 1000, (n_tasks, 512, 512, 512//8))
d_send = np.random.randint(0, 1000, (n_tasks, 512, 512, 512//8))
b = list(range(n_procs))
c = np.arange(n_procs*2).reshape(n_procs, -1)

t0 = time.time()

print('start main')
print('Scatter a')
comm.Scatter(a_send, None, root=MPI.ROOT)
comm.Scatter(d_send, None, root=MPI.ROOT)
# comm.scatter(b, root=0)
# comm.Bcast(c, root=MPI.ROOT)


# a = np.zeros((n_tasks, 512, 512, 512//8), int)
# d = np.zeros((n_tasks, 512, 512, 512//8), int)
# comm.Gather(None, a, root=MPI.ROOT)
# comm.Gather(None, d, root=MPI.ROOT)
# b = comm.gather(b, root=MPI.ROOT)
t1 = time.time()
print('cost', t1-t0, 's')
comm.Disconnect()







