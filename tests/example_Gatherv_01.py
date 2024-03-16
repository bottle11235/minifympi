from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# 每个进程创建不同数量的数据
send_data = np.array([rank] * (rank + 1), dtype=int)
print(f'rank{rank}', 'send_data', send_data)

# 确定每个进程发送的数据量
counts = comm.gather(len(send_data), root=0)

# 收集所有进程的数据
recv_data = None
if rank == 0:
    # 接收数据
    displacements = [sum(counts[:i]) for i in range(len(counts))]
    recv_data = np.zeros(sum(counts), dtype=int)
    print('recvbuf参数需要接受两个参数：')
    print([recv_data, counts])
    comm.Gatherv(send_data, [recv_data, counts], root=0)
else:
    comm.Gatherv(send_data, None, root=0)


if rank == 0:
    print("Gathered data:", recv_data)
