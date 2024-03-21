from typing import Any
from mpi4py import MPI
import numpy as np
import sys
import os
import re
import inspect
from textwrap import indent, dedent
import types
import black
import warnings
from functools import wraps, update_wrapper, reduce
import copy
from colorama import Fore, Back, Style
import traceback
from rich.syntax import Syntax
from rich.console import Console
from mpi4py.util.dtlib import from_numpy_dtype
from ..utils.code import get_source_with_requires, get_decorators
# from rich.theme import Theme
from itertools import chain
from functools import reduce
import uuid

#TODO
#- notebook模式报错处理
#  notebook模式下，调用mmp.Gather报错的话不会直接结束多进程，而是抛出报错，
#  多进程依然再等待消息。这样用户就不需要重复开启多进程了。
#- start_comm/close_comm检查是否已经启动/结束
#  如果已经启动/结束，就不会在采取任何行动，而是打印一个warning
#- notebook模式应当作为开发模式
#  我们应该把notebook当做是快捷开发平台，调用mmp.scatter['a'] = data的时候，将
#  data发送到root进程，然后在root进程执行mmp.scatter['a'] = data。这样可以保证
#  mpi的表现与mpirun完全一致的，缺点是内存开销将会是现在的2倍。

class MMPOptions:
    def __init__(self) -> types.NoneType:
        self.theme = 'lightbulb'
        self.is_show_code = True
        self.width = 100


class MinifyMPIBase:
    comm_cls = []

    def __init__(self, *args, **kwargs) -> None:

        #TODO 优化MinifyMPIBase
        #- n_procs如果是None，则根据comm.size来推断。
        self.comm = None
        self.comm_world = MPI.COMM_WORLD
        self.options = MMPOptions()
        self.console = Console()
        self.setup_comm_cls()

        self.gs = None          # 全局变量
        self.ls = {}            # 局部变量 


    @property
    def rank(self):
        return self.comm.rank

    @property
    def n_procs(self):
        return self.comm.size

    @property
    def is_main(self):
        return self.comm.rank == 0

    @property
    def is_root(self):
        return self.comm.rank == 0
    
    @property
    def ROOT(self):
        return 0

    @property
    def proc_id(self):
        return '(main)[proc0]' if self.is_main else f'[proc{self.rank}]'

    @classmethod
    def register_comm_cls(cls, comm_cls):
        cls.comm_cls.append(comm_cls)
        return comm_cls


    def setup_comm_cls(self):
        for cls in self.comm_cls:
            setattr(self, cls.__name__[3:], cls(self))


    def start_comm(self, gs=None):
        raise NotImplementedError
            

    def close_comm(self):
        resp = {'comm_type': 'exit'}
        self.all_send(resp)
        # self.comm.Disconnect()

    def all_send(self, resp):
        for i in range(self.n_procs):
            # self.log('all_send', resp)
            self.comm.send(resp, dest=i)

    def log(self, category, *msgs, color=None, ):
        #TODO - 使用原生的logger
        color = color if color else Fore.BLUE
        print(
            Fore.GREEN+f'{self.proc_id}'+Fore.RESET,
            color+f'[{category}]'+Fore.RESET,
            ' '.join([str(msg) for msg in msgs]),
        )


    def parallel(self, func, gs=None, ignores=None, requires=None):
        #NOTE notebook模式下，jit装饰的函数缺少__globals__属性，.py文件则
        #     不会缺少。因此，我们需要提供一个变量来解释gs参数。

        # MinifyMPI 类将会被忽略，这将是一个全局通用的变量
        gs = {} if gs is None else gs        
        gs.update(getattr(func, '__globals__', {}))
        ignores = [] if ignores is None else ignores
        alias = [key for key, value in gs.items() if value is self]
        ignores.extend(alias)


        # 为了避免递归调用，我们将MinifyMPI.parallel装饰器注释掉
        code = get_source_with_requires(func, gs, ignores, requires)
        decs = get_decorators(func)
        for dec_code, dec in decs.items():
            if dec in alias:
                code = re.sub(f'(\s*)(@{dec_code})', r'\1#\2', code)
        # self.log('parallel', '\n'+code)
        # self.log('parallel', ignores)


        self.exec(code)
        mpi_func = MPIFunction(self, func)
        setattr(self, func.__code__.co_name, mpi_func)
        update_wrapper(mpi_func, func)
        return func
    

    def __setitem__(self, keys, values):
        if isinstance(keys, str):
            keys = (keys, )

        if isinstance(values, MPIFunction):
            for key in keys:
                self.gs[key] = None
        
            arguments = values.send_arguments()
            code_args = ', '.join((chain(*arguments.values())))
            code_res = ', '.join([f'mmp.gs["{key}"]' for key in keys])
            code = f'{code_res} = {values.func.__code__.co_name}({code_args})'
            self.exec(code)


        else:
            '''bcast'''


    def __getitem__(self, keys):
        '''
        获取进程中的变量。这是一种标记符号，并不会放回进程里的数据。
        仅在并行函数内作为参数使用，以及删除进程内数据时使用。
        '''
        _keys = keys
        if isinstance(keys, str):
            keys = (keys,)
        elif not isinstance(keys, (list, tuple)):
            raise KeyError(keys)
        for key in keys:
            if key not in self.gs:
                raise KeyError(f'key `{key}` not Found.')
        vars = tuple(MPIVar(self, key) for key in keys)            
        if isinstance(_keys, str):
            return vars[0]
        else:
            return vars


class MPIFunction:
    def __init__(self, mmp, func):
        self.mmp = mmp
        self.func = func
        self.bound = {}


    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        sig = inspect.signature(self.func)
        bound = sig.bind(*args, **kwargs)
        self.bound['pargs'] = bound.arguments
        self.bound['args'] = {f'{i}': item for i, item in enumerate(bound.arguments.pop('args', {}))}
        self.bound['kwargs'] = bound.arguments.pop('kwargs', {})
        return self


    def send_arguments(self):
        arguments = {}
        for arg_type, dct in self.bound.items():
            arguments[arg_type] = []
            for key, value in dct.items():
                storage = 'gs' if isinstance(value, MPIVar) else 'ls'
                prefix = f'{key}=' if arg_type=='kwargs' else ''
                if isinstance(value, np.ndarray):
                    self.mmp.Bcast(storage='ls', **{key: value})
                elif not isinstance(value, MPIVar):
                    self.mmp.bcast(storage='ls', **{key: value})
                arguments[arg_type].append(f'{prefix}mmp.{storage}["{key}"]')
        return arguments


class MPIVar:
    '''
    TODO - MPIVar 的展示
    我们最好再Mpi类内维护一个metadata的字典，用于记录进程内变量的信息，
    通过MPIVar来标记变量时，就会展示相关的信息，例如类型，大小和传输类型
    等信息。
    '''
    def __init__(self, mmp, key) -> None:
        self.mmp = mmp
        self.key = key
    

    def __str__(self) -> str:
        return f"<<MPIVar> name='{self.key}'>"
    
    def __repr__(self) -> str:
        return self.__str__()


class MPICommBase:
    def __init__(self, mmp=None, ) -> None:
        self.mmp = mmp

    @property
    def is_main(self):
        return self.mmp.is_main
    
    @property
    def is_root(self):
        return self.mmp.is_root


    @property
    def comm(self):
        return self.mmp.comm
    
    @property
    def ROOT(self):
        return self.mmp.ROOT
    
    @property
    def gs(self):
        return self.mmp.gs
    
    @property
    def ls(self):
        return self.mmp.ls
    
    def log(self, category, *args, color=None):
        return self.mmp.log(category, *args, color)
    

    def exit(self):
        self.comm.bcast({'comm_type': 'exit'}, root=self.ROOT)

    def generate_resp(self, key, value, storage='gs'):
        resp = {
            'comm_type': self.__class__.__name__[3:],
            'name': key,
            'type': type(value).__name__,
            'dtype': str(getattr(value, 'dtype', '-')),
            'shape': getattr(value, 'shape', '-'),
            'storage': storage,
        }
        return resp


    def __main__(self, resp, senddata=None, recvdata=None, **kwargs):
        # self.comm.bcast(resp, root=self.ROOT)
        # self.log('__main__', resp)
        self.mmp.all_send(resp)

    def __comm__(self, senddata=None, recvdata=None, **kwargs):
        '''传输数据。所有进程都需要执行的通信代码。'''


@MinifyMPIBase.register_comm_cls
class MPIexec(MPICommBase):
    def __call__(self, code, *args: Any, **kwargs: Any) -> Any:
        resp = {'comm_type': 'exec', 'code': code}
        self.__main__(resp)
        self.__comm__(resp)


    def __comm__(self, resp):
        if not self.is_main or self.is_root:
            self.exec(resp)


    def exec(self, resp=None, **kwargs):
        try:
            fpath = f'{uuid.uuid4()}.py'
            with open(fpath, "w") as f:
                f.write(resp['code'])
            exec(compile(resp['code'], fpath, 'exec'), self.gs)
        except Exception as e:
            # 捕获错误并打印错误信息和行号
            self.log(f'{type(e).__name__}', f'{e}', color=Fore.RED)
            self.mmp.console.print_exception(
                theme=self.mmp.options.theme, 
                width=self.mmp.options.width
            )
        finally:
            if os.path.exists(fpath):
                os.remove(fpath)


class MPICommSetItem(MPICommBase):
    def __setitem__(self, key, value):
        if isinstance(key, str):
            kwargs = {key:value}
        if isinstance(key, tuple):
            kwargs = {k: v for k, v in zip(key, value)}
        # if isinstance(key, str):
        #     names = key,
        # if not isinstance(value, tuple):
        #     values = value,
        # self.log('__setitem__', names, values)

        self.__call__(**kwargs)
    
    
    def __call__(self, storage='gs', **kwargs: Any) -> None:
        for key, value in kwargs.items():
            resp = self.generate_resp(key, value, storage)
            self.__main__(resp)
            self.__comm__(resp, value)
    

    def __set_data__(self, key, value, storage='gs'):
        value = None if self.is_main and not self.is_root else value
        getattr(self, storage)[key] = value


@MinifyMPIBase.register_comm_cls
class MPIbcast(MPICommSetItem):
    def __comm__(self, resp=None, senddata=None, **kwargs):
        # self.log('__comm__', resp)
        recvdata = self.comm.bcast(senddata, root=self.ROOT)
        self.__set_data__(resp['name'], recvdata, resp['storage'])


@MinifyMPIBase.register_comm_cls
class MPIBcast(MPICommSetItem):
    def __comm__(self, resp=None, senddata=None, storage='gs'):
        senddata = np.zeros(resp['shape'], resp['dtype']) if senddata is None else senddata
        self.comm.Bcast(senddata, root=self.ROOT)
        self.__set_data__(resp['name'], senddata, resp['storage'])


@MinifyMPIBase.register_comm_cls
class MPIscatter(MPICommSetItem):
    def __comm__(self, resp=None, senddata=None, storage='gs'):
        recvdata = self.comm.scatter(senddata, root=self.ROOT)
        self.__set_data__(resp['name'], recvdata, resp['storage'])


@MinifyMPIBase.register_comm_cls
class MPIScatter(MPICommSetItem):
    def __comm__(self, resp=None, senddata=None, storage='gs'):
        recvdata = np.zeros((resp['shape'][0]//self.mmp.n_procs,)+resp['shape'][1:], resp['dtype'])
        self.comm.Scatter(senddata, recvdata, root=self.ROOT)
        self.__set_data__(resp['name'], recvdata, resp['storage'])


@MinifyMPIBase.register_comm_cls
class MPIScatterv(MPICommSetItem):
    #TODO Scatterv 优化 
    # - 数据类型检查
    #   mpi4py并非所有的numpy数据类型都支持，例如np.float16就不支持，可以考虑进行数据
    #   类型的检测
    # - 可以指定count、displ这些参数。
    #   Scatterv可以指定count（数量）、displ（偏移），我们应当支持接收这些参数。



    def __call__(self, storage='gs', count=None, displ=None, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            resp = self.generate_resp(key, value, storage)
            self.__main__(resp, value, count, displ)
            self.__comm__(resp, value)

    def __main__(self, resp, data, count=None, dspl=None):
        if count is None and dspl is None:
            task_count, count, displ = self.assign_tasks(data)
        resp['task_count'] = list(task_count)
        resp['count'] = list(count)
        resp['displ'] = list(displ)
        self.mmp.all_send(resp)


    def __comm__(self, resp, senddata=None):
        pass
        # self.log('__comm__', resp)
        shape = list(resp['shape'])
        shape[0] = resp['task_count'][self.mmp.rank]
        count = resp['count']
        displ = resp['displ']
        #NOTE - [WARNING] yaksa: 1 leaked handle pool objects 
        #  这个提示是由from_numpy_dtype抛出的，看起来不会带来任何错误。
        #LINK - https://github.com/firedrakeproject/firedrake/issues/2672
        datatype = from_numpy_dtype(resp['dtype'])
        recvdata = np.zeros(shape, resp['dtype'])
        self.comm.Scatterv([senddata, count, displ, datatype], recvdata, root=self.ROOT)
        self.__set_data__(resp['name'], recvdata, resp['storage'])
        self.comm.Barrier()


    def assign_tasks(self, data):
        axis=0
        if self.mmp.n_procs > data.shape[axis]:
            raise ValueError(f'`n_tasks`({data.shape[axis]}) must be greater than' 
                            f'`n_procs`({self.mmp.n_procs})')
        task_count = np.ones(self.mmp.n_procs, int)*data.shape[axis]//self.mmp.n_procs
        task_count[:data.shape[axis]%self.mmp.n_procs] += 1
        count = task_count*np.prod(data.shape)/data.shape[axis]
        displ = np.zeros_like(count)
        displ[1:] = np.cumsum(count)[:-1]
        return task_count, count, displ


class MPICommGetItem(MPICommBase):
    def __getitem__(self, key):
        if isinstance(key, str):
            key = (key, )
        elif isinstance(key, tuple):
            pass
        return self.__call__(*key)


    def __call__(self, *args: Any, storage='gs', **kwargs: Any) -> None:
        returns = []
        for key in args:
            resp = self.generate_resp(key, None, storage)
            self.__main__(resp)
            data = self.__comm__(resp)
            returns.append(data)
        return returns[0] if len(returns) == 1 else returns


@MinifyMPIBase.register_comm_cls
class MPIgather(MPICommGetItem):
    def __comm__(self, resp=None):
        # self.log('gather', resp)
        return self.comm.gather(getattr(self, resp['storage'])[resp['name']], root=self.ROOT)


@MinifyMPIBase.register_comm_cls
class MPIGather(MPICommGetItem):
    def __main__(self, resp):
        # 发送元数据，包括shape、type、dtype
        resp['status'] = 'check_data'
        self.mmp.all_send(resp)
        resps = self.comm.gather(resp, root=self.ROOT)
        if resps[0]['type'] == 'NoneType':
            resp.update(self.generate_resp(resp['name'], getattr(self, resp['storage'])[resp['name']], resp['storage']))
            resps[0] = resp
        else:
            resp.update(resps[0])


        # resps包含了所有的metadata，如果metadata不是完全一样，
        # 报错并通知次进程停止Gather
        if not all([item == resp for item in resps]):
            self.mmp.all_send({'status': 'stop'})
            msg = 'Cannot Gather data for different metadata.\n'
            msg += '\n'.join([f'sub{i}: {meta}' for i, meta in enumerate(resps)])
            raise ValueError(msg)
        
        resp['shape'] = (resp['shape'][0]*self.mmp.n_procs,) + resp['shape'][1:]
        resp['status'] = 'continue'
        self.mmp.all_send(resp)
    

    def __comm__(self, resp):
        if resp['status'] == 'check_data':
            senddata = getattr(self, resp['storage'])[resp['name']]
            resp = self.generate_resp(resp['name'], senddata, resp['storage'])
            resp['status'] = 'check_data'
            self.comm.gather(resp, root=self.ROOT)
        elif resp['status'] == 'continue':
            recvdata = np.zeros(resp['shape'], resp['dtype']) if self.is_main else None
            self.comm.Gather(getattr(self, resp['storage'])[resp['name']], recvdata, root=self.ROOT)
            return recvdata


@MinifyMPIBase.register_comm_cls
class MPIGatherv(MPICommGetItem):
    def __main__(self, resp):
        resp['status'] = 'check_data'
        self.mmp.all_send(resp)
        resps = self.comm.gather(resp, root=self.ROOT)
        if resps[0]['type'] == 'NoneType':
            resp.update(self.generate_resp(resp['name'], getattr(self, resp['storage'])[resp['name']], resp['storage']))
            resps[0] = resp
        else:
            resp.update(resps[0])

        # 根据元数据，检查数据是否有效
        if not all([resp['type'] == 'ndarray' for resp in resps]) or\
           not all([resp['dtype'] == resps[0]['dtype'] for resp in resps]) or\
           not all([resp['shape'][1:] == resps[0]['shape'][1:] for resp in resps]):
            self.comm.bcast({'status': 'stop'}, root=self.ROOT)
            msg = 'Cannot Gatherv data. The metadata is shown below:\n'
            msg += '\n'.join([f'sub{i}: {meta}' for i, meta in enumerate(resps)])
            raise ValueError(msg)
        
        resp['status'] = 'continue'        
        resp['count'] = [np.prod(resp['shape']) for resp in resps]
        resp['shape'] = reduce(lambda x, y: (x[0]+y[0], *x[1:]), 
                       [resp['shape'] for resp in resps])
        self.mmp.all_send(resp)


    def __comm__(self, resp=None, **kwargs):
        if resp['status'] == 'check_data':
            senddata = getattr(self, resp['storage'])[resp['name']]
            resp = self.generate_resp(resp['name'], senddata, resp['storage'])
            self.comm.gather(resp, root=self.ROOT)
        
        elif resp['status'] == 'continue':
            # self.log('Gatherv resp', resp)
            recvdata = np.zeros(resp['shape'], resp['dtype']) if self.is_main else None
            self.comm.Gatherv(getattr(self, resp['storage'])[resp['name']], [recvdata, resp['count']], root=self.ROOT)
            return recvdata
        

#SECTION - non-blocking communication
#TODO 支持非阻塞通信
#  下面的代码虽然是通过非阻塞通信函数来通信，但是使用了wait函数，因此不是非阻塞的。
#  可以考虑通过异步来实现非阻塞通信，例如，发送数据时，isend之后直接结束，在接收端
#  通过异步方法来判断是不是接收完成，接收完成后再将数据保存起来。
class MPICommISetItem(MPICommBase):
    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key)==2:
            idxs, names = key
        else:
            raise KeyError(key)
        
        idxs = idxs,
        names = names,
        values = value,
        self.__call__(idxs, names, values)
    

    def __call__(self, idxs=None, names=None, values=None, storage='gs', **kwargs: Any) -> types.NoneType:
        for idx, name, value in zip(idxs, names, values):
            if idx == 0 and self.is_root:
                self.__set_data__(name, value, storage)
                continue

            resp = self.generate_resp(str(name), value, storage)
            resp['dest'] = int(idx)
            resp['source'] = 0
            self.__main__(resp)
            self.__comm__(resp, value)


    def __set_data__(self, key, value, storage='gs'):
        value = None if self.is_main and not self.is_root else value
        getattr(self, storage)[key] = value

    def __main__(self, resp, senddata=None, recvdata=None, **kwargs):
        self.comm.send(resp, dest=resp['dest'])


@MinifyMPIBase.register_comm_cls
class MPIisend(MPICommISetItem):
    def __comm__(self, resp, senddata=None, recvdata=None):
        if self.mmp.rank == resp['dest']:
            data = self.comm.irecv(source=resp['source']).wait()
            self.__set_data__(resp['name'], data, resp['storage'])


        elif self.mmp.rank == resp['source']:
            self.comm.isend(senddata, resp['dest'])


class MPICommIGetItem(MPICommBase):
    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key)==2:
            idxs, names = key
        else:
            raise KeyError(key)
        idxs = idxs,
        names = names,
        return self.__call__(idxs, names)

    def __call__(self, idxs=None, names=None, storage='gs', **kwargs: Any) -> types.NoneType:
        returns = []
        for idx, name in zip(idxs, names):
            if idx == 0 and self.is_root:
                data = getattr(self, storage)[name]
            else:
                resp = self.generate_resp(str(name), None, storage)
                resp['dest'] = 0
                resp['source'] = idx
                self.__main__(resp)
                data = self.__comm__(resp)
            returns.append(data)
        return returns[0] if len(returns) == 1 else returns

    
    def __main__(self, resp, senddata=None, recvdata=None, **kwargs):
        self.comm.send(resp, dest=resp['source'])


@MinifyMPIBase.register_comm_cls
class MPIirecv(MPICommIGetItem):
    def __comm__(self, resp, senddata=None, recvdata=None):
        if self.mmp.rank == resp['dest']:
            return self.comm.irecv(source=resp['source']).wait()

        elif self.mmp.rank == resp['source']:
            senddata = getattr(self, resp['storage'])[resp['name']]
            self.comm.isend(senddata, resp['dest']).wait()


@MinifyMPIBase.register_comm_cls
class MPIiloc(MPICommSetItem):
    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            raise KeyError(key)
        idxs, names = np.array(key[0]), np.array(key[1])
        idxs = idxs.reshape(1) if len(idxs.shape) == 0 else idxs
        names = names.reshape(1) if len(names.shape) == 0 else names
        
        if isinstance(value, np.ndarray):
            values = (value,)
        if isinstance(value, int):
            values = (value,)

        if idxs.shape[0] == 1 and names.shape[0] > 1:
            idxs = idxs.repeat(names.shape[0])
        elif idxs.shape[0] > 1 and names.shape[0] == 1:
            names = names.repeat(idxs.shape[0])
        
        self.__call__(idxs, names, values)            
            
    
    def __call__(self, idxs=None, names=None, values=None, storage='gs', **kwargs: Any) -> types.NoneType:
        for idx, name, value in zip(idxs, names, values):
            if idx == 0 and self.is_root:
                self.__set_data__(name, value, storage)

            resp = self.generate_resp(str(name), value, storage)
            resp['dest'] = int(idx)
            resp['source'] = 0
            self.__main__(resp)
            # self.__comm__(resp, value)



    def __main__(self, resp, senddata=None, recvdata=None, **kwargs):
        self.log('__main__', resp)
        self.comm.send(resp, dest=resp['dest'])


    def __comm__(self, resp, senddata=None, recvdata=None):
        self.log('__comm__', resp)

        if self.is_main:
            self.log('__comm__', resp)
            if resp['type'] == 'ndarray':
                self.comm.Isend(senddata, dest=resp['dest'])
            else:
                self.comm.isend(senddata, dest=resp['dest'])

        else:
            self.log('__comm__', resp)
            if resp['type'] == 'ndarray':
                recvdata = np.zeros(resp['shape'], resp['dtype'])
                self.comm.Irecv(recvdata, source=resp['source'])
            else:
                pass
                self.comm.irecv(recvdata, source=resp['source'])
        
        self.__set_data__(resp['name'], recvdata, resp['storage'])


#!SECTION


@MinifyMPIBase.register_comm_cls
class MPIsend(MPICommISetItem):
    def __comm__(self, resp, senddata=None, recvdata=None):
        if self.mmp.rank == resp['dest']:
            data = self.comm.recv(source=resp['source'])
            self.__set_data__(resp['name'], data, resp['storage'])


        elif self.mmp.rank == resp['source']:
            self.comm.send(senddata, resp['dest'])


@MinifyMPIBase.register_comm_cls
class MPISend(MPICommISetItem):
    def __comm__(self, resp, senddata=None, recvdata=None):
        if self.mmp.rank == resp['dest']:
            recvdata = np.zeros(resp['shape'], resp['dtype'])
            self.comm.Recv(recvdata, source=resp['source'])
            self.__set_data__(resp['name'], recvdata, resp['storage'])


        elif self.mmp.rank == resp['source']:
            self.comm.Send(senddata, resp['dest'])


@MinifyMPIBase.register_comm_cls
class MPIrecv(MPICommIGetItem):
    def __comm__(self, resp, senddata=None, recvdata=None):
        if self.mmp.rank == resp['dest']:
            return self.comm.recv(source=resp['source'])

        elif self.mmp.rank == resp['source']:
            senddata = getattr(self, resp['storage'])[resp['name']]
            self.comm.send(senddata, resp['dest'])


@MinifyMPIBase.register_comm_cls
class MPIRecv(MPICommIGetItem):
    def __comm__(self, resp, senddata=None, recvdata=None):
        if self.mmp.rank == resp['dest']:
            resp0 = self.comm.recv(source=resp['source'])
            recvdata = np.zeros(resp0['shape'], resp0['dtype'])
            self.comm.Recv(recvdata, source=resp['source'])
            return recvdata

        elif self.mmp.rank == resp['source']:
            resp0 = self.generate_resp(resp['name'], getattr(self, resp['storage'])[resp['name']], resp['storage'])
            self.comm.send(resp0, dest=resp['dest'])
            senddata = getattr(self, resp['storage'])[resp['name']]
            self.comm.Send(senddata, resp['dest'])

