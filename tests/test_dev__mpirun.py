import numpy as np
import sys
sys.path.append('/mnt/d/Python/minifympi')
from minifympi.dev import MinifyMPI


mmp = MinifyMPI(4, 10)
mmp.start_comm()

# mmp.bcast['a'] = 1
# a = mmp.gather['a']
# mmp.log('a', a)


# data = np.arange(6).reshape(2, 3)
# mmp.Bcast['b'] = data
# b = mmp.Gather['b']
# mmp.log('b', b)

# 测试scatter
# mmp.scatter['a'] = list(range(mmp.n_procs))
# a = mmp.gather['a']
# mmp.log('a', a)

# # 测试Scatter
# data = np.arange(mmp.n_procs*3).reshape(mmp.n_procs, -1)
# mmp.Scatter['a'] = data
# a = mmp.Gather['a']
# mmp.log('a', a)


# code = 'mmp.log("exec", mmp.gs["b"])'
# mmp.exec(code)

# # mmp.log('tasks_count', mmp.tasks_count)
# mmp.log("mmp.gs['a']", mmp.gs['a'])


# data = np.random.randint(0, 100, (15, 3))
data = np.arange(36).reshape(4, -1)

# mmp.Scatterv['a'] = data
mmp.Scatterv(a=data)
code = 'mmp.log("exec", mmp.gs["a"])'
mmp.exec(code)
mmp.log('data', data)


mmp.close_comm()

