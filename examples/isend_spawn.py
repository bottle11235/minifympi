from mpi4py import MPI
import sys
import numpy as np

comm = MPI.COMM_SELF.Spawn(sys.executable,
                           args=['/mnt/d/Python/minifympi/examples/isend_spawn_sub.py'],
                           maxprocs=4)
rank = comm.Get_rank()
data = {'rank': 0, 'a': 7, 'b': 3.14, 'd': np.array([1, 2, 3])}
for i in range(4):
    req = comm.isend(data, dest=i)

comm.Disconnect()