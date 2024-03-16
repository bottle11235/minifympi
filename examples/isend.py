from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    data = {'rank': 0, 'a': 7, 'b': 3.14}
    for i in range(comm.size):
        req = comm.isend(data, dest=i)
        # req.wait()
else:
    req = comm.irecv(source=0)
    data = req.wait()
    print(f'rank{rank}', data)