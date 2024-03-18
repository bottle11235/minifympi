from mpi4py import MPI
import numpy as np
import time


comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()


req = comm.irecv(source=0)
data = req.wait()
print(f'rank{rank}', data)
comm.Disconnect()
