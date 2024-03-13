from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

sendbuf = None
if rank == 0:
    sendbuf = np.empty([size, 3], dtype='i')
    sendbuf.T[:,:] = range(size)
recvbuf = np.empty(3, dtype='i')
comm.Scatter(sendbuf, recvbuf, root=0)
assert np.allclose(recvbuf, rank)

# if rank == 0:
print(f'rank{rank}', recvbuf)
data = recvbuf

if rank == 1:
    recvbuf = np.empty([size, 3], dtype='i')
else:
    recvbuf = None
comm.Gather(data, recvbuf, root=1)
if rank == 1:
    print(f'rank{rank}', recvbuf)


