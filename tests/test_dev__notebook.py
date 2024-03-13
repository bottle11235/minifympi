import numpy as np
import sys
sys.path.append('/mnt/d/Python/minifympi')
from minifympi.dev import MinifyMPINB


mmp = MinifyMPINB(4, 10)
mmp.start_comm()

# 测试bcast
mmp.bcast['a'] = [1]
mmp.log('a', mmp.gather['a'])

# 测试Bcast
a = np.array([[1, 2, 3], [4, 5, 6]])
mmp.Bcast['a'] = a
mmp.log('a', mmp.Gather['a'])


# 测试scatter
mmp.scatter['a'] = [1, 2, 3, 4]
mmp.log('a', mmp.gather['a'])

# 测试Scatter
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
mmp.Scatter['a'] = a
mmp.log('a', mmp.Gather['a'])

mmp.log("mmp.gs['a']", mmp.gs['a'])


# 测试 exec
code = 'mmp.log("exec", mmp.gs["a"])'
mmp.exec(code)


# 测试Scatterv
data = np.arange(36, dtype=np.float32).reshape(6, -1)
mmp.Scatterv['a'] = data
# mmp.Scatterv(a=data)
code = 'mmp.log("exec", mmp.gs["a"])'
mmp.exec(code)
mmp.log('data', data)


mmp.close_comm()