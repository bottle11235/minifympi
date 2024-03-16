import numpy as np
import sys
sys.path.append('/mnt/d/Python/minifympi')
from minifympi.core.notebook import MinifyMPI
from numba import jit

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

# mmp.log('gs', mmp.gs)


# b = mmp.Gather['b']
# mmp.log('b', b)

# # 测试scatter
# mmp.scatter['a'] = list(range(mmp.n_procs))
# a = mmp.gather['a']
# mmp.log('a', a)
# code = 'mmp.log("exec", mmp.gs["a"])'
# mmp.exec(code)
# mmp.log('gs', mmp.gs)

# # 测试Scatter
# data = np.arange(mmp.n_procs*3).reshape(mmp.n_procs, -1)
# mmp.Scatter['a'] = data
# a = mmp.Gather['a']
# mmp.log('Scatter a', a)


# code = 'mmp.log("exec", mmp.gs["a"])'
# mmp.exec(code)


# # data = np.random.randint(0, 100, (15, 3))
# data = np.arange(35).reshape(-1, 5)
# # mmp.Scatterv['a'] = data
# mmp.Scatterv(a=data)
# code = 'mmp.log("exec", mmp.gs["a"])'
# mmp.exec(code)
# mmp.log('data', data)

# mmp.bcast['a'] = 1
# mmp.bcast['b'] = 1
# a = mmp.gather['a']
# mmp.log('a', a)
# code = 'mmp.log("exec", mmp.gs["a"])'
# mmp.exec(code)

# # mmp.gs['a'] = 3
# print(mmp['a'])
# print(mmp['a', 'b'])

# mmp.bcast['a'] = 3

# mma = mmp

# def test1(func):
#     return func


# def test2(n_procs=None):
#     def decorator(func):
#         return func
#     return decorator


# # @jit
# # @test1
# @test2(3)
# @mmp.parallel
# def test(a, b, ):
#     print('test', a, b)
#     return a, b

# code = ''' 
# test(2, 4)
# '''
# mmp.exec(code)

# mmp['res0', 'res1'] = mmp.test(mmp['a'], np.array([1, 2]))
# code = 'mmp.log("ls", mmp.ls)\nmmp.log("gs", mmp.gs.keys())'
# mmp.exec(code)
# mmp.log('main gs', mmp.gs.keys())
# mmp.log('res0', mmp.gather['res0'])
# code = 'mmp.log("res", mmp.gs["res0"], mmp.gs["res1"])'
# mmp.exec(code)

# 测试保存到ls字典
# mmp.bcast(a=1, storage='ls')

# b = np.array([1, 2])
# mmp.Bcast(b=b, storage='ls')


# mmp.scatter(c=list(range(mmp.n_procs)), storage='ls')

# data = np.arange(mmp.n_procs*2).reshape(mmp.n_procs, -1)
# mmp.Scatter(d=data, storage='ls')

# data = np.arange(35).reshape(-1, 5)
# mmp.Scatterv['a'] = data

# code = 'mmp.log("ls", mmp.ls)'
# mmp.exec(code)

# mmp.log('ls', mmp.ls)
# mmp.log('gs', mmp.gs)


# mmp.bcast(a=1, storage='ls')
# mmp.bcast(b=np.array([1, 2]), storage='ls')


########################
#parallel

# code = '''
# import numpy as np
# mmp.ls["_returns_"] = (1, np.array([1, 2]))

# meta = {}
# for i, value in enumerate(mmp.ls['_returns_']):
#     key = f'_return{i}_'
#     mmp.ls[key] = value
#     if isinstance(value, np.ndarray):
#         # value.shape % mmp.n_procs
#         meta[key] = mmp.Gather.generate_resp(key, value, 'ls')
#     else:
#         meta[key] = mmp.gather.generate_resp(key, value, 'ls')
# mmp.ls['_meta_'] = meta
# '''
# mmp.exec(code)

# mmp.ls['_meta_'] = None
# _meta_ = mmp.gather('_meta_', storage='ls')
# returns = []
# for key, meta in _meta_[0].items():
#     mmp.ls[meta['name']] = None
#     if meta['comm_type'] == 'gather':
#         returns.append(mmp.gather(meta['name'], storage='ls'))
#     if meta['comm_type'] == 'Gather':
#         returns.append(mmp.Gather(meta['name'], storage='ls'))

# mmp.log('returns', returns)


# 测试Gatherv
data = np.arange(20).reshape(-1, 2)
mmp.Scatterv['a'] = data

# code = 'mmp.log("Scatterv", mmp.gs["a"])'
# mmp.exec(code)
a = mmp.Gatherv['a']
mmp.log('a', a)


mmp.close_comm()

# from minifympi.utils.code import find_requires

# print(find_requires(test))