import sys
sys.path.append("/mnt/d/Python/minifympi/")

from minifympi.core import MinifyMPI
from mpi4py import MPI
import numpy as np


n_tasks = 12
n_procs = 5
mmp = MinifyMPI(n_procs, n_tasks, sys.argv[1] if len(sys.argv)==2 else 'dynamic')


mmp.start_comm({'mmp': mmp})


a = list(range(mmp.n_tasks))
b = np.arange(2*n_tasks).reshape(n_tasks, -1)

mmp.Scatter['a'] = a
mmp.Scatter['b'] = b

@mmp.parallel
def test(a, b):
    a/0
    return a, b
mmp.test['res1', 'res2'](mmp['a'], mmp['b'])


res1, res2 = mmp.Gather['res1', 'res2']
print(res1, res2)


# code = '''
# def test():
#     2/0
# a = 3
# b = 4
# test()
# # 1/0
# '''

# resp = {
#     'comm_type': 'exec',
#     'code': code,
# }
# mmp.comm.bcast(resp, root=mmp.root)
# mmp.exec(resp=resp)


mmp.close_comm()





