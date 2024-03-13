from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

if rank == 0:
    sendbuf = np.arange(24.).reshape(8, -1)

    # count: the size of each sub-task
    ave, res = divmod(sendbuf.size, nprocs)
    count = [ave + 1 if p < res else ave for p in range(nprocs)]
    count = np.array(count)

    # displacement: the starting index of each sub-task
    displ = [sum(count[:p]) for p in range(nprocs)]
    displ = np.array(displ)

    print('count0', count)
    print('displ0', displ)

    n_procs = nprocs
    data = sendbuf
    count = np.ones(n_procs, int)*data.size//n_procs
    count[:data.size%n_procs] += 1

    displ = np.zeros_like(count)
    displ[1:] = np.cumsum(count)[:-1]
    print('count1', count)
    print('displ1', displ)



else:
    sendbuf = None
    # initialize count on worker processes
    count = np.zeros(nprocs, dtype=int)
    displ = None

# broadcast count
comm.Bcast(count, root=0)

# initialize recvbuf on all processes
recvbuf = np.zeros(count[rank], float).reshape(-1, 3)


comm.Scatterv([sendbuf, count, displ], recvbuf, root=0)

print('After Scatterv, process {} has data:'.format(rank), recvbuf)