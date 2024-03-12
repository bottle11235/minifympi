from mpi4py import MPI
import numpy as np
import time


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_procs = comm.size
n_tasks = 8

def test(a:'S', b:'s', c=None, d:'S'=None, *args, **kwargs)->'G,g,g':
    print(a[0, 0, 0, :3])
    return a, b, d

a_send = None
d_send = None
b = None
if rank == 0:
    a_send = np.random.randint(0, 1000, (n_tasks, 1024, 1024, 512//8))
    d_send = np.random.randint(0, 1000, (n_tasks, 1024, 1024, 512//8))
    b = list(range(n_procs))
    c = np.arange(n_procs*2).reshape(n_procs, -1)
else:
    c = np.zeros((n_procs, 2), int)

a_recv = np.zeros((n_tasks//n_procs, 1024, 1024, 512//8), int)
d_recv = np.zeros((n_tasks//n_procs, 1024, 1024, 512//8), int)

if rank == 0:
    t0 = time.time()
comm.Scatter(a_send, a_recv, root=0)
comm.Scatter(d_send, d_recv, root=0)
b = comm.scatter(b)
comm.Bcast(c, root=0)

a_recv, b, d_recv = test(a_recv, b, c, d_recv, 5, 10, 11)

if rank == 0:
    a = np.zeros((n_tasks, 1024, 1024, 512//8), int)
    d = np.zeros((n_tasks, 1024, 1024, 512//8), int)
else:
    a = None
    d = None

comm.Gather(a_recv, a, root=0)
comm.Gather(d_recv, d, root=0)
b = comm.gather(b, root=0)

if rank == 0:
    t1 = time.time()
    print('cost', t1-t0, 's')
