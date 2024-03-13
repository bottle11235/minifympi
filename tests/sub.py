from mpi4py import MPI
import numpy as np
import time

comm = MPI.Comm.Get_parent()
size = comm.Get_size()
rank = comm.Get_rank()
n_procs = comm.size
n_tasks = 8
print(f'start sub{rank}')

a_recv = np.zeros((n_tasks//n_procs, 512, 512, 512//8), int)
d_recv = np.zeros((n_tasks//n_procs, 512, 512, 512//8), int)
c = np.zeros((n_procs, 2), int)


comm.Scatter(None, a_recv, root=0)
comm.Scatter(None, d_recv, root=0)
# b = comm.scatter(None, root=0)
# comm.Bcast(c, root=0)

print(f'sub{rank}', a_recv[0, 0, 0, 0])

if rank == 0:
    data = comm.recv(source=0)
    print(f'sub{rank}', data)
    print('send data to main')
    comm.send([4, 5, 6], dest=0)

# if rank == 0
# def test(a:'S', b:'s', c=None, d:'S'=None, *args, **kwargs)->'G,g,g':
#     print(a[0, 0, 0, :3])
#     return a, b, d
# a_recv, b, d_recv = test(a_recv, b, c, d_recv, 5, 10, 11)

# comm.Gather(a_recv, None, root=0)
# comm.Gather(d_recv, None, root=0)
# b = comm.gather(b, root=0)

comm.Disconnect()
