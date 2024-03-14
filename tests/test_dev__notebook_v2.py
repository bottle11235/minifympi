import numpy as np
import sys
sys.path.append('/mnt/d/Python/minifympi')
from minifympi.core.notebook import MinifyMPI


mmp = MinifyMPI(4, 10)
mmp.start_comm()

mmp.bcast['a'] = 1
a = mmp.gather['a']
mmp.log('a', a)
code = 'mmp.log("exec", mmp.gs["a"])'
mmp.exec(code)



data = np.arange(6).reshape(1, -1)
mmp.Bcast['b'] = data
code = 'mmp.log("exec", mmp.gs["b"])'
mmp.exec(code)

mmp.log('gs', mmp.gs)


b = mmp.Gather['b']
mmp.log('b', b)

# 测试scatter
mmp.scatter['a'] = list(range(mmp.n_procs))
a = mmp.gather['a']
mmp.log('a', a)
code = 'mmp.log("exec", mmp.gs["a"])'
mmp.exec(code)
mmp.log('gs', mmp.gs)

# 测试Scatter
data = np.arange(mmp.n_procs*3).reshape(mmp.n_procs, -1)
mmp.Scatter['a'] = data
a = mmp.Gather['a']
mmp.log('Scatter a', a)


code = 'mmp.log("exec", mmp.gs["a"])'
mmp.exec(code)


# data = np.random.randint(0, 100, (15, 3))
data = np.arange(35).reshape(-1, 5)
# mmp.Scatterv['a'] = data
mmp.Scatterv(a=data)
code = 'mmp.log("exec", mmp.gs["a"])'
mmp.exec(code)
mmp.log('data', data)


mmp.close_comm()

