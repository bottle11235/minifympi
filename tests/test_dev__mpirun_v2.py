#%%
import numpy as np
import sys
sys.path.append('/mnt/d/Python/minifympi')
from minifympi.core.base import MinifyMPI
from numba import jit

#%%
mmp = MinifyMPI(4, 10)
mmp.start_comm()

# mmp.bcast['a'] = 1
# a = mmp.gather['a']
# mmp.log('a', a)
# code = 'mmp.log("exec", mmp.gs["a"])'
# mmp.exec(code)



# data = np.arange(6).reshape(1, -1)
# mmp.Bcast['b'] = data
# code = 'mmp.log("exec", mmp.gs["b"])'
# mmp.exec(code)

# b = mmp.Gather['b']
# mmp.log('b', b)

# # 测试scatter
# mmp.scatter['a'] = list(range(mmp.n_procs))
# a = mmp.gather['a']
# mmp.log('a', a)
# code = 'mmp.log("exec", mmp.gs["a"])'
# mmp.exec(code)
# # mmp.log('gs', mmp.gs)

# # 测试Scatter
# data = np.arange(mmp.n_procs*3).reshape(mmp.n_procs, -1)
# mmp.Scatter['a'] = data
# a = mmp.Gather['a']
# mmp.log('Scatter a', a)


# code = 'mmp.log("exec", mmp.gs["a"])'
# mmp.exec(code)


# # data = np.random.randint(0, 100, (15, 3))
# data = np.arange(36).reshape(4, -1)
# mmp.Scatterv['a'] = data
# # mmp.Scatterv(a=data)
# code = 'mmp.log("exec", mmp.gs["a"])'
# mmp.exec(code)
# # mmp.log('data', data)



# mmp.bcast['a'] = 1
# a = mmp.gather['a']
# mmp.log('a', a)
# code = 'mmp.log("exec", mmp.gs["a"])'
# mmp.exec(code)

# # mmp.gs['a'] = 3
# print(mmp['a'])




# mmp.bcast['a'] = 3

# # mma = mmp

# @jit
# @mmp.parallel
# def test(a, b):
#     print('test', a, b)
#     return a, b

# code = ''' 
# test(2, 4)
# '''
# mmp.exec(code)

# mmp['res0', 'res1'] = mmp.test(mmp['a'], np.array([1, 2]))
# code = 'mmp.log("ls", mmp.ls)'
# mmp.exec(code)
# # code = 'mmp.log("res", mmp.gs["res0"], mmp.gs["res1"])'
# # mmp.exec(code)
# mmp.log('res0', mmp.gather['res0'])


# 测试Gatherv
data = np.arange(20).reshape(-1, 2)
mmp.Scatterv['a'] = data
# code = 'mmp.log("Scatterv", mmp.gs["a"])'
# mmp.exec(code)
a = mmp.Gatherv['a']
mmp.log('a', a)

mmp.close_comm()

