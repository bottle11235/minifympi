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
import builtins

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
        self.comm.bcast(resp, root=self.ROOT)
        self.comm.Disconnect()


    def log(self, category, *msgs, color=None, ):
        #TODO - 使用原生的logger
        color = color if color else Fore.BLUE
        print(
            Fore.GREEN+f'{self.proc_id}'+Fore.RESET,
            color+f'[{category}]'+Fore.RESET,
            ' '.join([str(msg) for msg in msgs]),
        )

    
    def generate_script(self, func):
        ''''''
        code_str = '#'
        return code_str

    def parallel(self, func):
        pass

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
                #TODO 根据具体环境，检查是否运行Disconnect函数
                self.comm.Disconnect()
                exit()
            else:
                getattr(self, resp['comm_type']).__comm__(resp=resp)

    def extract_decorate_names(self,source_code):
        decorator_pattern = re.compile(r'@(\w+)')
        decorators = decorator_pattern.findall(source_code.split('def')[0])
        return decorators

    def get_dependent_code(self, func, seen=None, codes=None, modules=None):
        if seen is None:
            seen = set()
        if codes is None:
            codes = []
        if modules is None:
            modules = set()
        
        if func in seen:
            return codes, modules
        seen.add(func)
        if hasattr(func, '__wrapped__'):
            source_code = inspect.getsource(func.__wrapped__)
            decorator_names= self.extract_decorator_names(source_code)
            for decorator_name in decorator_names:
                if globals()[decorator_name].__module__ == "__main__" and globals()[decorator_name] not in seen:
                    self.get_dependent_code(globals()[decorator_name], seen, codes, modules)
                else:
                    modules.add("from " + globals()[decorator_name].__module__ + " import " + decorator_name)
            while hasattr(func, '__wrapped__'):
                func = func.__wrapped__
        if func.__module__ != "__main__":
            modules.add("from " + func.__module__ + " import " + func.__name__)
        else:
            try:
                source = inspect.getsource(func)
                codes.append(source)
            except Exception as e:
                print(f"Error getting source of {func.__name__}: {e}")

        module = inspect.getmodule(func)  
        if not module:
            pass
        else:
            func_globals = func.__globals__
            
            for name in func.__code__.co_names:
                if name in dir(builtins):
                    continue

                if name in func_globals and not name.startswith("__"):
                    obj = func_globals[name]
                    if isinstance(obj, types.ModuleType):
                        modules.add(f"import {obj.__name__} as {name}" if obj.__name__ != name else f"import {obj.__name__}")
                    elif hasattr(obj, '__module__'):
                        if obj.__module__ != "__main__":
                            modules.add(f"from {obj.__module__} import {name}")

        for name in func.__code__.co_names:
            try:
                if hasattr(globals()[name], '__wrapped__'):
                    self.get_dependent_code(globals()[name], seen, codes, modules)
            except:
                pass

            obj =  globals().get(name, None) 
            if inspect.isfunction(obj) and obj not in seen:
                self.get_dependent_code(obj, seen, codes, modules)
        
        return codes, modules


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
        self.comm.bcast(resp, root=self.ROOT)

    def __comm__(self, senddata=None, recvdata=None, **kwargs):
        '''传输数据。所有进程都需要执行的通信代码。'''


@MinifyMPI.register_comm_cls
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
            kwargs = {ikey: ivalue for ikey, ivalue in zip(key, value)}
        self.__call__(**kwargs)
    
    
    def __call__(self, storage='gs', **kwargs: Any) -> None:
        for key, value in kwargs.items():
            resp = self.generate_resp(key, value, storage)
            self.__main__(resp)
            self.__comm__(resp, value)
    

    def __set_data__(self, key, value, storage='gs'):
        value = None if self.is_main and not self.is_root else value
        getattr(self, storage)[key] = value


@MinifyMPI.register_comm_cls
class MPIbcast(MPICommSetItem):
    def __comm__(self, resp=None, senddata=None, **kwargs):
        recvdata = self.comm.bcast(senddata, root=self.ROOT)
        self.__set_data__(resp['name'], recvdata)


@MinifyMPI.register_comm_cls
class MPIBcast(MPICommSetItem):
    def __comm__(self, resp=None, senddata=None):
        senddata = np.zeros(resp['shape'], resp['dtype']) if senddata is None else senddata
        self.comm.Bcast(senddata, root=self.ROOT)
        self.__set_data__(resp['name'], senddata)


@MinifyMPI.register_comm_cls
class MPIscatter(MPICommSetItem):
    def __comm__(self, resp=None, senddata=None):
        recvdata = self.comm.scatter(senddata, root=self.ROOT)
        self.__set_data__(resp['name'], recvdata)


@MinifyMPI.register_comm_cls
class MPIScatter(MPICommSetItem):
    def __comm__(self, resp=None, senddata=None):
        recvdata = np.zeros((resp['shape'][0]//self.mmp.n_procs,)+resp['shape'][1:], resp['dtype'])
        self.comm.Scatter(senddata, recvdata, root=self.ROOT)
        self.__set_data__(resp['name'], recvdata)


@MinifyMPI.register_comm_cls
class MPIScatterv(MPICommSetItem):
    #TODO Scatterv 优化 
    # - 数据类型检查
    #   mpi4py并非所有的numpy数据类型都支持，例如np.float16就不支持，可以考虑进行数据
    #   类型的检测
    # - 可以指定count、displ这些参数。
    #   Scatterv可以指定count（数量）、displ（偏移），我们应当支持接收这些参数。
    #FIXME - [WARNING] yaksa: 1 leaked handle pool objects
    #   目前会出现[WARNING] yaksa: 1 leaked handle pool objects提示，要找出这个问题
    #   并解决。

    def __call__(self, storage='gs', count=None, displ=None, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            resp = self.generate_resp(key, value, storage)
            self.__main__(resp, value, count, displ)
            self.__comm__(resp, value)

    def __main__(self, resp, data, count=None, dspl=None):
        if count is None and dspl is None:
            task_count, count, displ = self.assign_tasks(data)
        
        resp['task_count'] = task_count
        resp['count'] = count
        resp['displ'] = displ
        self.comm.bcast(resp, root=self.ROOT)


    def __comm__(self, resp, senddata=None):
        shape = list(resp['shape'])
        shape[0] = resp['task_count'][self.mmp.rank]
        count = resp['count']
        displ = resp['displ']
        datatype = from_numpy_dtype(resp['dtype'])
        recvdata = np.zeros(shape, resp['dtype'])
        
        self.comm.Scatterv([senddata, count, displ, datatype], recvdata, root=self.ROOT)
        # self.log('Scatterv resp', resp)
        self.__set_data__(resp['name'], recvdata)


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
            resp = self.generate_resp(key, None)
            self.__main__(resp)
            data = self.__comm__(resp)
            returns.append(data)
        return returns[0] if len(returns) == 1 else returns


@MinifyMPI.register_comm_cls
class MPIgather(MPICommGetItem):
    def __comm__(self, resp=None):
        return self.comm.gather(self.gs[resp['name']], root=self.ROOT)


@MinifyMPI.register_comm_cls
class MPIGather(MPICommGetItem):
    def __main__(self, resp):
        # 发送元数据，包括shape、type、dtype
        resp['status'] = 'check_data'
        self.comm.bcast(resp, root=self.ROOT)
        resps = self.comm.gather(resp, root=self.ROOT)
        if resps[0]['type'] == 'NoneType':
            resp.update(self.generate_resp(resp['name'], self.gs[resp['name']]))
            resps[0] = resp
        else:
            resp.update(resps[0])

        # resps包含了所有的metadata，如果metadata不是完全一样，
        # 报错并通知次进程停止Gather
        if not all([item == resp for item in resps]):
            self.comm.bcast({'status': 'stop'}, root=self.ROOT)
            msg = 'Cannot Gather data for different metadata.\n'
            msg += '\n'.join([f'sub{i}: {meta}' for i, meta in enumerate(resps)])
            raise ValueError(msg)
        
        resp['shape'] = (resp['shape'][0]*self.mmp.n_procs,) + resp['shape'][1:]
        resp['status'] = 'continue'
        self.comm.bcast(resp, root=self.ROOT)
    

    def __comm__(self, resp):
        if resp['status'] == 'check_data':
            senddata = self.gs[resp['name']]
            resp = self.generate_resp(resp['name'], senddata)
            resp['status'] = 'check_data'
            self.comm.gather(resp, root=self.ROOT)
        elif resp['status'] == 'continue':
            recvdata = np.zeros(resp['shape'], resp['dtype']) if self.is_main else None
            self.comm.Gather(self.gs[resp['name']], recvdata, root=self.ROOT)
            return recvdata


