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

# from rich.theme import Theme
import uuid

class MMPOptions:
    def __init__(self) -> types.NoneType:
        self.theme = 'lightbulb'
        self.is_show_code = True
        self.width = 100


class MinifyMPIBase:
    comm_cls = []

    def __init__(self, n_procs=2, n_tasks=2) -> None:

        #TODO 优化MinifyMPIBase
        #- n_procs如果是None，则根据comm.size来推断。
        #- n_tasks似乎并不需要。
        self.n_procs = n_procs
        self.n_tasks = n_tasks

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
    def is_main(self):
        return self.comm.rank == 0
    
    @property
    def ROOT(self):
        return 0

    @property
    def proc_id(self):
        return 'main' if self.is_main else f'sub{self.rank}'

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
        self.comm.bcast(resp, root=self.ROOT)
        self.comm.Disconnect()


    def log(self, category, *msgs, color=None, ):
        #TODO - 使用原生的logger
        color = color if color else Fore.BLUE
        print(
            Fore.GREEN+f'[{self.proc_id}]'+Fore.RESET, 
            color+category+Fore.RESET,
            ' '.join([str(msg) for msg in msgs]),
        )


class MinifyMPI(MinifyMPIBase):    
    def start_comm(self, gs=None):
        self.gs = gs if gs is not None else {'mmp': self}
        # self.assign_tasks()
        self.comm = MPI.COMM_WORLD
        if self.is_main:
            return
        
        while True:
            resp = self.comm.bcast(None, root=self.ROOT)
            if resp['comm_type'] == 'exit':
                self.comm.Disconnect()
                exit()
            else:
                getattr(self, resp['comm_type'])._sub(resp=resp)


class MinifyMPINB(MinifyMPIBase):    
    @property
    def is_main(self):
        return self.comm.Get_parent() == MPI.COMM_NULL

    @property
    def ROOT(self):
        return MPI.ROOT if self.is_main else 0
    

    def start_comm(self, gs=None):
        self.gs = gs if gs is not None else {'mmp': self}
        # self.assign_tasks()

        #STUB - 删除临时库
        # 开发完成就去掉临时库的部分
        code = [
            'import sys, os',
            'sys.path.append("/mnt/d/Python/minifympi/")',
            'from minifympi.dev import MinifyMPINB',
            f'mmp = MinifyMPINB({self.n_procs}, {self.n_tasks})',
            'mmp.start_comm()',
        ]
        code = '\n'.join(code)
        fpath = '/mnt/d/Python/minifympi/tmp_subprocss.py'
        with open(fpath, 'w') as f:
            f.write(code)

        if MPI.Comm.Get_parent() == MPI.COMM_NULL:
            self.comm = MPI.COMM_SELF.Spawn(sys.executable,
                args=[fpath], maxprocs=self.n_procs)
            return
        
        self.comm = MPI.Comm.Get_parent()
        while True:
            resp = self.comm.bcast(None, root=self.ROOT)
            if resp['comm_type'] == 'exit':
                self.comm.Disconnect()
                exit()
            else:
                getattr(self, resp['comm_type'])._sub(resp=resp)


class MPICommBase:
    def __init__(self, mmp=None, ) -> None:
        self.mmp = mmp

    @property
    def is_main(self):
        return self.mmp.is_main
    

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
    
    def log(self, category=None, msg=None):
        return self.mmp.log(category, msg)
    

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


@MinifyMPI.register_comm_cls
class MPIexec(MPICommBase):
    def __call__(self, code, *args: Any, **kwargs: Any) -> Any:
        resp = {'comm_type': 'exec', 'code': code}
        self._main(resp)

    def _main(self, resp):
        self.comm.bcast(resp, root=self.ROOT)
        self.exec(resp)

    def _sub(self, resp):
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
            kwargs = {ikey: ivalue for ikey, ivalue in zip(key, value)}
        self.__call__(**kwargs)
    
    
    def __call__(self, storage='gs', **kwargs: Any) -> None:
        for key, value in kwargs.items():
            resp = self.generate_resp(key, value, storage)
            self._main(resp, value)


@MinifyMPI.register_comm_cls
class MPIbcast(MPICommSetItem):
    def _main(self, resp=None, data=None, **kwargs):
        resp['data'] = data
        self.comm.bcast(resp, root=self.ROOT)
        #FIXME - notebook模式，主进程不需要写入数据
        # 在mpirun模式下，无论是主进程还是次进程，都要写入数据，但是
        # notebook模式下，主进程不需要写入数据。
        self.gs[resp['name']] =  data


    def _sub(self, resp=None, **kwargs):
        getattr(self, resp['storage'])[resp['name']] = resp['data']


@MinifyMPI.register_comm_cls
class MPIBcast(MPICommSetItem):
    def _main(self, resp, data, **kwargs):
        self.comm.bcast(resp, root=self.ROOT)
        self.comm.Bcast(data, root=self.ROOT)
        self.gs[resp['name']] = data


    def _sub(self, resp=None):
        data = np.zeros(resp['shape'], resp['dtype'])
        self.comm.Bcast(data, root=self.ROOT)
        getattr(self, resp['storage'])[resp['name']] = data


@MinifyMPI.register_comm_cls
class MPIscatter(MPICommSetItem):
    def _main(self, resp, data):
        self.comm.bcast(resp, root=self.ROOT)
        self.gs[resp['name']] =  self.comm.scatter(data, root=self.ROOT)

    def _sub(self, resp):
        self.gs[resp['name']] =  self.comm.scatter(None, root=self.ROOT)


@MinifyMPI.register_comm_cls
class MPIScatter(MPICommSetItem):
    def _main(self, resp, data):
        self.comm.bcast(resp, root=self.ROOT)
        shape = (resp['shape'][0]//self.mmp.n_procs,) + resp['shape'][1:]
        data_recv = np.zeros(shape, resp['dtype'])
        self.comm.Scatter(data, data_recv, root=self.ROOT)
        self.gs[resp['name']] =  data_recv


    def _sub(self, resp):
        data = np.zeros((resp['shape'][0]//self.mmp.n_procs,)+resp['shape'][1:], resp['dtype'])
        self.comm.Scatter(None, data, root=self.ROOT)
        self.gs[resp['name']] =  data


@MinifyMPI.register_comm_cls
class MPIScatterv(MPICommSetItem):
    #TODO Scatterv 优化 
    # - 数据类型检查
    #   mpi4py并非所有的numpy数据类型都支持，例如np.float16就不支持，可以考虑进行数据
    #   类型的检测
    # - 提供axis参数
    #   axis只接受-1和0两个参数，-1表示按照Scatterv的默认行为进行分发，axis为0表示按照
    #   第一个轴进行数据分发。暂不支持更多的axis设定。
    # - 可以指定count、displ这些参数。
    #   Scatterv可以指定count（数量）、displ（偏移），我们应当支持接收这些参数。
    #FIXME - [WARNING] yaksa: 1 leaked handle pool objects
    #   目前会出现[WARNING] yaksa: 1 leaked handle pool objects提示，要找出这个问题
    #   并解决。
    def _main(self, resp, data, count=None, dspl=None):
        if count is None and dspl is None:
            task_count, count, displ = self.assign_tasks(data)
        
        resp['task_count'] = task_count
        self.comm.bcast(resp, root=self.ROOT)
        shape = list(data.shape)
        shape[0] = task_count[0]
        data_recv = np.zeros(shape, data.dtype)
        self.comm.Scatterv([data, count, displ, from_numpy_dtype(data.dtype)],
                            data_recv, root=self.ROOT)
        self.gs[resp['name']] = data_recv


    def __call__(self, storage='gs', count=None, displ=None, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            resp = self.generate_resp(key, value, storage)
            self._main(resp, value, count, displ)


    def _sub(self, resp):
        shape = list(resp['shape'])
        shape[0] = resp['task_count'][self.mmp.rank]
        data_recv = np.zeros(shape, resp['dtype'])
        self.comm.Scatterv(None, data_recv, root=self.ROOT)
        self.gs[resp['name']] = data_recv


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


    def __call__(self, *args: Any, **kwargs: Any) -> None:
        returns = []
        for key in args:
            resp = self.generate_resp(key, self.gs[key])
            data = self._main(resp)
            returns.append(data)
        return returns[0] if len(returns) == 1 else returns


@MinifyMPI.register_comm_cls
class MPIgather(MPICommGetItem):
    def _main(self, resp=None):
        self.comm.bcast(resp, root=self.ROOT)
        data = self.comm.gather(self.gs[resp['name']], root=self.ROOT)
        return data


    def _sub(self, resp=None):
        self.comm.gather(self.gs[resp['name']], root=self.ROOT)


@MinifyMPI.register_comm_cls
class MPIGather(MPICommGetItem):
    def _main(self, resp):
        data0, shape = self.gs[resp['name']], resp['shape']
        if not isinstance(data0, np.ndarray):
            raise TypeError

        # 发送元数据，包括shape、type、dtype
        self.comm.bcast(resp, root=self.ROOT)
        resps = self.comm.gather(resp, root=self.ROOT)

        # resps包含了所有的metadata，如果metadata不是完全一样，
        # 报错并通知次进程停止Gather
        if not all([item == resp for item in resps]):
            self.comm.bcast({'status': 'stop'}, root=self.ROOT)
            msg = 'Cannot Gather data for different metadata.\n'
            msg += '\n'.join([f'sub{i}: {meta}' for i, meta in enumerate(resps)])
            raise ValueError(msg)
        
        self.comm.bcast({'status': 'continue'}, root=self.ROOT)
        shape = (shape[0]*self.mmp.n_procs,) + shape[1:]
        data = np.zeros(shape, data0.dtype)
        self.comm.Gather(self.gs[resp['name']], data, root=self.ROOT)
        return data


    def _sub(self, resp):
        data = self.gs[resp['name']]
        self.comm.gather(self.generate_resp(resp['name'], data), root=self.ROOT)
        resp = self.comm.bcast(None, root=self.ROOT)
        if resp['status'] == 'continue':
            self.comm.Gather(data, None, root=self.ROOT)










