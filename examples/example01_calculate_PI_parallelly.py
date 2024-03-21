import time
import numpy as np
import sys, os
from mpi4py import MPI

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
comm = MPI.COMM_WORLD
if comm.size == 1:
    from minifympi.decorators.notebook import parallel
else:
    from minifympi.decorators.mpirun import parallel
n_procs = 4 if comm.size == 1 else None


def generate_data(num_samples):
    data = np.random.rand(num_samples, 2)
    return data


@parallel(n_procs=n_procs)
def monte_carlo_pi(data:"Sv") -> 'g':
    return 4*sum((data**2).sum(axis=1) < 1)


if __name__ == "__main__":
    n_tasks = int(1024*1024*1024/8/2)
    data = generate_data(n_tasks)
    time_start =  time.perf_counter()
    res = monte_carlo_pi(data)
    print(res)
    print(sum(res)/n_tasks)
    time_end = time.perf_counter()
    print(f"Time elapsed: {time_end - time_start} seconds")

