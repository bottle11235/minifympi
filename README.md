这个包的目的是简化mpi4py的使用。使得你只需要了解少量的并行计算的知识，就可以快速地开始计算了。
MinifyMPI目前尚处于开发阶段，预计稳定版将会与目前版本有很大的区别。

# 快速上手
对于MinifyMPI，最简单的使用方式是使用它提供的装饰器。假设你有一个形状为（1000，10）的numpy数组，想快速地对它的第1个轴求和。
你可以使用mpi4py，将这个数组拆分为10个（100，10）的数组，沿第一个轴求和：
```python
# ./my_sum.py
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 准备数据
if rank == 0:
    data = np.random.rand(1000, 10)
    result = np.zeros(1000, 'float64')
else:
    data = None
    result = None

recv_data = np.zeros(1000//size, 'float64')
comm.Scatter(data, recv_data, root=0)

# 定义并行计算函数
def my_sum(data):
    return data.sum(axis=1)

# 计算并收集计算结果
result_i = my_sum(recv_data)
comm.Gather(result_i, result, root=0)
```
然后在终端指定用10个进程进行计算，执行以下指令开始计算：
```bash
mpirun -np 10 python ./my_sum.py
```


如果使用MinifyMPI，能大幅减少数据通信的代码，
```python
# ./my_sum.py
from minifympi.wrappers import parallel
import numpy as np

# 准备数据
data = np.random.rand(1000, 10)

# 定义并行计算函数
@parallel(n_procs=10, n_tasks=1000)
def my_sum(data:'S'):
    return data.sum(axis=1)

# 计算并收集计算结果
result = my_sum(data)
```
我们使用`parallel`装饰器，指定使用10个进程（`n_procs=10`）来计算，通过函数注解（function annotation，`data:'S'`）的方式来指出`data`需要分发（`Scatter`）到每个进程。上述代码可以直接在`notebook`内运行，也可以在终端运行：
```bash
python ./my_sum.py
```


# parallel装饰器
`parallel`装饰器使用`mpi4py`的阻塞通信来传输数据，因此你需要了解`scatter`、`Scatter`、`bcast`、`Bcast`、`gather`和`Gather`这几个并行计算的关键概念。我们使用函数注解的方式来指示哪些数据按照哪种方式来进行传输。

- s: scatter
- S: scatter
- b: bcast
- B: Bcast
- g: gather
- G: Gather

如果你的参数没有函数注解，那么默认情况下会对python object使用bcast，对numpy.ndarray使用Bcast的方式来传输数据，对于计算结果（如果有的话），会分别使用gather和Gather来将python object和numpy.ndarray的进行收集。

## parallel装饰器的约束
我们对parallel装饰器做了一些约束，要求通过Scatter、Gather传输的数据其长度必须等于`n_tasks`。对于上面的例子，如果你已经设定了`n_tasks`为1000，而你希望把一个（2000. 100）的数据平均分发到10个进程，这是不被允许的，你需要先将数组`reshape`为（1000，2，100），此时数组的长度（`len(data)`）就会与任务数相等，然后再分发数据。

# MinifyMPI class
我们提供了一个名为`MinifyMPI`的类，以便通过更加python风格的方式来进行数据通信。
```python
from minifympi.core import MinifyMPI
import numpy as np

n_procs=5
n_tasks=13
mmp = MinifyMPI(n_procs, n_tasks, mode='dynamic')
mmp.start_comm()

# 广播数据
mmp.bcast['data1'] = 3
mmp.Bcast['data2'] = np.arange(3)

# 分发数据
mmp.scatter['data3'] = list(range(n_procs))
mmp.Scatter['data4'] = np.arange(n_tasks*3).reshape(n_tasks, 3)

# 收集数据
data1 = mmp.gather['data1']
data2 = mmp.Gather['data2']
assert data1 == list(range(n_procs))
assert data2 == np.arange(n_tasks*3).reshape(n_tasks, 3)

# 发送数据到指定进程
mmp.loc[1, 'data5'] = [1, 2]                             # 发送到1号进程
mmp.loc[2, 'data6'] = np.arange(3)                       # 发送到2号进程
mmp.loc[[1, 3, 4], 'data7'] = np.arange(3)               # 发送到1、3和4号进程
mmp.loc[[2:], 'data8'] = 'hello, world!'                 # 发送到2至5号进程
mmp.loc[2, ['data9', 'data10']] = 1, [2]                 # 同时发送多个数据到2号进程

# 从指定进程接收数据
data5 = mmp.loc[1, 'data5']                             # 接收1号进程的data5
data6 = mmp.loc[2, 'data6']                             # 接收2号进程的data6
data7 = mmp.loc[[1, 3, 4], 'data7']                     # 接收1、3和4号进程的data7
data8 = mmp.loc[[2:], 'data8']                          # 接收2至5号进程的data8
data9, data10 = mmp.loc[2, ['data9', 'data10']]         # 接收2号进程的data9和data10

assert data5 == [1, 2]
assert data6 == np.arange(3)
assert data7 == [np.arange(3)]*3
assert data8 == ['hello, world!']*3
assert data9 == 1
assert data10 == [2]

```

`MinifyMPI`类提供了一个`parallel`装饰器，可以将函数用于并行计算。
```python
@mmp.parallel
def test(a, b, c, *args, **kwargs):
    '''do something'''
    return a, b, c

a = np.arange(n_tasks*3).reshape(n_tasks, 3)
b = list(range(n_procs))
c = 10

# 分发数据
mmp.Scatter['a'] = a
mmp.scatter['b'] = b

# 运行并行计算函数。函数的返回值会保存在每个进程里，我们指定结果保存到名为
# `res0`、`res1`和`res2`的变量里。
mmp.test['res0', 'res1', 'res2'](mmp['a'], mmp['b'], c)
# 指定进程执行并行计算，这一个不一定会支持。
mmp.test[1, ['res0', 'res1', 'res2']](mmp['a'], mmp['b'], c)
mmp.test[[1, 2], ['res0', 'res1', 'res2']](mmp['a'], mmp['b'], c)
mmp.test[:, ['res0', 'res1', 'res2']](mmp['a'], mmp['b'], c)
mmp.test[-1, ['res0', 'res1', 'res2']](mmp['a'], mmp['b'], c)


res1, res2 = mmp.gather['res1', 'res2']
res0 = mmp.Gather['res0']
assert res0 == a
assert res1 == b
assert res2 == [c]*n_procs
```




# TODO
- 介绍n_procs和n_tasks的概念
- 举例介绍每种传输方式
- 介绍MinifyMPI类的使用，一种简单的通信方式，还有它的限制，目前只能实现阻塞通信，非阻塞看情况开发。
- 将求和的例子改为1000个向量和1000种不同的旋转矩阵相乘的例子。
    这样的例子可以更好地表现并行带来的优势，也可以说明n_tasks和数据长度的关系。
- MinifyMPI类提供他自己的rank和send、recv等通信函数。
    因为MinifyMPI是通过`spawn`来动态生成进程的，所有rank的表现和静态的并不一样。我们
    希望提供MinifyMPI自己的通信函数，来让不熟悉多进程的用户可以获得像使用静态多进程那样的体验。
    也就是0号进程是主（根）进程，其他进程是子进程。
- 对读写文件的支持。目前还不支持。

对于并行函数，我们有以下方案：
```python
@mmp.parallex
def test(a, b):
    return a + b
mmp['res'] = test(*mmp[['a', 'b']])


@mmp.parallex(outputs=['res'])
def test(a, b):
    return a + b
test(*mmp[['a', 'b']])


@mmp.parallex
def test(a, b):
    return a + b
mmp.test['res'](*mmp[['a', 'b']])


def test(a, b):
    return a + b
mmp.parallex(test, inputs=mmp[['a', 'b']], outputs=mmp['res'])
```
理论上，以上4种都可以实现，第一种虽然最直观，但是要进行2次通信，效率最低。第二种是最简单的，但是outputs被固定了，最死板。第三种很灵活，但是需要用户习惯新的逻辑，第四种也很直观、很灵活，但是代码上比较冗余。
我比较喜欢后面两种，可以考虑同时支持。
