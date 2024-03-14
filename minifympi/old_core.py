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
# from rich.theme import Theme
import uuid

# from rich import traceback as rich_traceback
# rich_traceback.install()

#NOTE - 我们提供了两种模式：
# 1. dynamic
# 2. mpirun
# 分别对应了notebook和超算命令行两种使用情况。（也许超算也支持dynamic?
# 通过mpirun启动后，除根进程外其他进程直接exit，根进程生成子进程。)
# 这两种方式的rank是不统一的，dynamic的子进程是从0开始的，而mpirun我们
# 人为规定0进程是主进程，其他是子进程。

#TODO - MinifyMPI拆分
# 把MinifyMPI拆分成两个类，分别对应上述的两种模式。让用户根据具体情况，
# 选择恰当的类。这能极大优化代码，减少判断模式的代码。

#TODO - 报错
# 如果子进程出错了，我们如何让子进程继续处于待命状态？而不是要重启所有工作？
# 对于数据传递，一种方法是尽可能在主进程进行数据检查，确保能正确地传递数据。对于gather，
# 可以先把变量名发送给子进程，子进程检查有没有这个变量，有的话，就发简报给主
# 进程，表示没问题，然后主进程接收数据。
# 函数并行计算则比较麻烦。
# 如果有一种通用的方法就最好了。

# TODO 临时文件的路径
# 我们应该提供一个参数，可以指定临时文件保存路径
class MMPOptions:
    def __init__(self) -> types.NoneType:
        self.theme = 'lightbulb'
        self.is_show_code = True
        self.width = 100



class MinifyMPI:
    comm_cls = []

    def __init__(self, n_procs=2, n_tasks=2, mode='dynamic') -> None:

        #TODO n_procs如果是None，则根据comm.size来推断。
        self.n_procs = n_procs
        self.n_tasks = n_tasks
        self.mode = mode
        self.comm = None
        self.options = MMPOptions()
        self.console = Console()

        # 开发阶段，还是手动注册吧
        # self.bcast = MPIbcast(self)
        # self.Bcast = MPIBcast(self)
        # self.scatter = MPIscatter(self)
        # self.Scatter = MPIScatter(self)
        # self.gather = MPIgather(self)
        # self.Gather = MPIGather(self)
        # self.loc = MPIloc(self)

        # 通过装饰器把类注册到一个字典，然后自动生成一些的代码。
        # 开发结束后，通过以下方法来自动注册
        self.setup_comm_cls()

        self.gs = None
        self.ls = {}


    @property
    def rank(self):
        if self.mode == 'dynamic':
            if self.is_main:
                return 0
            else:
                return self.comm.rank + 1
        else:
            return self.comm.rank

    @property
    def is_main(self):
        if self.mode == 'mpirun':
            return self.comm.rank == 0
        else:
          return self.comm.Get_parent() == MPI.COMM_NULL

    @property
    def root(self):
        if self.mode == 'mpirun':
            return 0
        else:
            if self.is_main:
                return MPI.ROOT
            else:
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


    def assign_tasks(self, **kwargs):
        #TODO 可以手动指定。初始化的时候提供n_procs和n_tasks参数
        # 在assig_tasks提供n_tasks_main和n_tasks_sub参数，则可以为进程
        # 指定任务数
        if self.n_procs > self.n_tasks:
            raise ValueError(f'`n_tasks`({self.n_tasks}) must be greater than' 
                            f'`n_procs`({self.n_procs})')
        self.n_tasks_sub = self.n_tasks // self.n_procs
        self.n_tasks_main = self.n_tasks - (self.n_procs-1)*self.n_tasks_sub


    def close_comm(self):
        # self.log('exit gs', self.gs)
        resp = {'comm_type': 'exit'}
        self.comm.bcast(resp, root=self.root)
        self.comm.Disconnect()


    def log(self, category=None, msg=None, color=None, *args):
        #TODO - 使用原生的logger
        color = color if color else Fore.BLUE
        print(
            Fore.GREEN+f'[{self.proc_id}]'+Fore.RESET, 
            color+category+Fore.RESET,
            ' '.join([str(msg)] + [str(item) for item in args]),
        )


    def start_comm(self, gs=None):
        self.gs = gs if gs else {'mmp': self}
        self.assign_tasks()
        #STUB - 删除临时库
        # 开发完成就去掉临时库的部分
        code = [
            'import sys, os',
            'sys.path.append("/mnt/d/Python/minifympi/")',
            'from minifympi.core import MinifyMPI',
            f'mmp = MinifyMPI({self.n_procs}, {self.n_tasks})',
            'mmp.start_comm(globals())',
        ]
        code = '\n'.join(code)
        with open('/mnt/d/Python/minifympi/tmp_subprocss.py', 'w') as f:
            f.write(code)

        if self.mode == 'dynamic':
            if MPI.Comm.Get_parent() == MPI.COMM_NULL:
                self.comm = MPI.COMM_SELF.Spawn(sys.executable,
                        args=['/mnt/d/Python/minifympi/tmp_subprocss.py'],
                        maxprocs=self.n_procs-1)
                return
            else:
                self.comm = MPI.Comm.Get_parent()
        elif self.mode == 'mpirun':
            self.comm = MPI.COMM_WORLD
            if self.is_main:
                return
        while True:
            resp = self.comm.bcast(None, root=self.root)
            # self.log('resp', resp)
 
            if resp['comm_type'] == 'exit':
                # self.log('exit gs', self.gs)
                self.comm.Disconnect()
                exit()
            else:
                # self.log('resp', resp)
                getattr(self, resp['comm_type'])(resp=resp)


    def find_requires(self, func, g=None, ignores=None):
        ''' 
        Find dependency libraries and functions for `func`.

        Parameters
        ----------
        func : function
            Any function.
        '''
        ignores = ignores if ignores else ["__builtins__", "__file__"]
        g = g if g else dict(func.__globals__)
        requires = self.find_requires_code(func, g, ignores)
        return {require: g[require] for require in requires if require in g}


    def find_requires_code(self, func, g, ignores=None):
        requires = [
            name for name in func.__code__.co_names
            if name not in ignores \
            and name != func.__code__.co_name \
            and hasattr(sys.modules['__main__'], name)
        ]
        for item in requires:
            if item in g and isinstance(g[item], types.FunctionType):
                for key in self.find_requires(g[item], ignores + requires):
                    if key not in requires:
                        requires.append(key)
        return requires
    

    def find_decorator_requires(self, func):
        # NOTE - 不支持class类型的装饰器
        # class类型的装饰器需要初始化，我们不知道如何初始化，因此不支持class类型的
        # 装饰器
        # TODO
        # 如果有class类型的装饰器，我们应当弹出提示，表示不支持。
        code = dedent(inspect.getsource(func))
        decorators = [line.strip() for line in code.split('\n') if line.startswith('@')]
        decorators = [re.findall('@(.+'+('?)\(' if '(' in line else ')'), line)[0] for line in decorators]
        requires_decorators = self.find_requires(eval('lambda: ({})'.format(','.join(decorators))), func.__globals__)
        return requires_decorators
    

    def generate_script(self, func):
        '''
        创建依赖函数，包括引入相关的库和被调用的函数等。 值得注意的是，如果函数内部使用了全局变量，
        将会抛出一个存在风险的提示。虽然我们有办法提取全局变量，但是没有必要。
        '''
        # 根据函数更新依赖库和依赖函数
        requires = self.find_requires(func)
        requires_decorators = self.find_decorator_requires(func)
        requires.update(requires_decorators)
        
        code_import = [
        ]
        code_func = []
        
        for key, value in requires.items():
            if isinstance(value, types.ModuleType):
                code_import.append(
                    f'import {key}' if key == value.__name__ else f'import {value.__name__} as {key}'
                )
            elif isinstance(value, types.FunctionType):
                if value.__module__ == '__main__':
                    code_func.append(inspect.getsource(value))
                else:
                    # f'import {key}' if key == value.__name__ else f'import {value.__name__} as {key}'
                    print(key == value.__name__.split('.')[-1])
                    code_import.append(f'from {value.__module__} import {key}' if key == value.__name__.split('.')[-1] else f'import {value.__module__} as {key}')
            elif key != 'mmp':
                warnings.warn(f'`{key}` is a global variable, which has the risk of being undefined.', UserWarning)
        
        code_func += [inspect.getsource(func)]
        code = code_import + code_func
        code = dedent('\n'.join(code))
        return black.format_str(code, mode=black.Mode())
    

    def parallel(self, func):
        if not self.is_main:
            return func
        
        # 将函数注册到每个进程
        code = self.generate_script(func)
        resp = {
            'comm_type': 'exec',
            'code': code,
            'co_name': func.__code__.co_name,
        }
        self.comm.bcast(resp, root=self.root)
        mpi_func = MPIFunction(self, func)
        setattr(self, func.__code__.co_name, mpi_func)
        update_wrapper(mpi_func, func)
        return func


    def exec(self, **kwargs):
        code = kwargs['resp']['code']
        #TODO - 展示代码错误
        # + 如果是文本格式的代码，报错时无法展示具体某行代码的细节，因此我们
        # 应当将code保存到临时文件，exec来执行该临时文件，然后报错。
        # + 这个临时文件每个进程都有不同名字，以免其他进程无法访问，或者复写
        # 内容。
        # - 如何能打印才是的错误信息？
        # self.log('exec', code)
        random_uuid = uuid.uuid4()
        fpath = f'{random_uuid}.py'

        try:
            # fpath = self.options.temp_fpath
            with open(fpath, "w") as f:
                f.write(code)
            exec(compile(code, fpath, 'exec'), self.gs)

        except Exception as e:
            # 捕获错误并打印错误信息和行号
            self.log(f'{type(e).__name__}', f'{e}', color=Fore.RED)
            
            # syntax = Syntax(
            #     code, "python", 
            #     theme=self.options.theme, 
            #     line_numbers=True
            # )
            # self.console.print(syntax, width=self.options.width)
            self.console.print_exception(
                theme=self.options.theme, 
                width=self.options.width
            )

        finally:
            if os.path.exists(fpath):
                os.remove(fpath)


    def __getitem__(self, key):
        '''
        获取进程中的变量。这是一种标记符号，并不会放回进程里的数据。
        仅在并行函数内作为参数使用，以及删除进程内数据时使用。

        TODO - 对于获取进程内多个变量的支持
        我们也许要添加同时获取多个变量的支持，实现起来也不困难，而且也
        符合python的使用习惯。
        '''
        if isinstance(key, str):
            if key not in self.gs:
                raise KeyError(f'key `{key}` not Found.')
        # elif isinstance(key, tuple):
        #     for item in key:
        #         if not isinstance(item, str):
        #             raise TypeError('only support str.')            
        #         if key not in self.gs:
        #             raise KeyError(f'key `{key}` not Found.')
        return MPIVar(self, key)
    

    def __delitem__(self, key):
        '''
        删除所有进程内名为`key`的变量。
        我们要求希望这个删除的逻辑也与pandas尽量一致，让用户尽量减少
        学习成本。

        删除一个变量：
        del mmp['a']

        删除多个变量：
        del mmp['a'], mmp['b']

        无法通过以下方法删除多个变量
        del mmp['a', 'b']
        del mmp[['a', 'b']]
        '''

        # 删除子进程变量
        resp = {
            'comm_type': '_del_',
            'name': key,
        }
        self.comm.bcast(resp, root=self.root)
        self._del_(resp=resp)


    def _del_(self, **kwargs):
        # REVIEW 删除数据
        # 目前支持全局删除，通过 del mmp['a']可以删除所有进程中的`a`变量。
        resp = kwargs['resp']
        if resp['name'] in self.gs:
            del self.gs[resp['name']]
        pass





class MPIFunction:
    def __init__(self, mmp, func):
        self.mmp = mmp
        self.func = func
        self.res_name = None


    def __getitem__(self, key):
        #TODO MPIFunction 的强化和完善
        # - 返回结果与变量名对不上
        # - 如果是一个生成器，我们也应当支持。
        
        # 这时候要报错。
        if not isinstance(key, (str, tuple)):
            raise KeyError(f'`key` accepts only `str` and `tuple`, '
               'but `{type(key).__name__}` was given.')
        if isinstance(key, tuple):
            for item in key:
                if not isinstance(item, str):
                    raise KeyError(f'`key` accepts only `str` and `tuple`, '
                       'but `{type(key).__name__}` was given.')
        self.res_name = key
        return self.record


    def record(self, *args, **kwargs):
        '''record'''
        sig = inspect.signature(self.func)
        bound = sig.bind(*args, **kwargs)
        metadata = {'pargs':{}, 'args':{}, 'kwargs':{}}

        for key, value in bound.arguments.items():
            if key in ['args', 'kwargs']:
                continue
            if isinstance(value, MPIVar):
                metadata['pargs'][key] = {'comm_type': 'mpivar'}
            elif isinstance(value, np.ndarray):
                metadata['pargs'][key] = {'comm_type': 'Bcast'}
            else:
                metadata['pargs'][key] = {'comm_type': 'bcast'}
        if 'args' in bound.arguments:
            for key, value in enumerate(bound.arguments['args']):
                key = str(key)
                if isinstance(value, MPIVar):
                    metadata['args'][key] = {'comm_type': 'mpivar'}
                elif isinstance(value, np.ndarray):
                    metadata['args'][key] = {'comm_type': 'Bcast'}
                else:
                    metadata['args'][key] = {'comm_type': 'bcast'}
        if 'kwargs' in bound.arguments:
            for key, value in bound.arguments['kwargs'].items():
                if isinstance(value, MPIVar):
                    metadata['kwargs'][key] = {'comm_type': 'mpivar'}
                elif isinstance(value, np.ndarray):
                    metadata['kwargs'][key] = {'comm_type': 'Bcast'}
                else:
                    metadata['kwargs'][key] = {'comm_type': 'bcast'}

        arguments = {'pargs': {}, 'args': {}, 'kwargs': {}}
        sub_arguments = {'pargs': {}, 'args': {}, 'kwargs': {}}
        for item in ['pargs', 'args', 'kwargs']:
            for i, (key, meta) in enumerate(metadata[item].items()):
                if meta['comm_type'] == 'mpivar':
                    arguments[item][key] = self.mmp.gs[key]
                    sub_arguments[item][key] = f'mmp.gs["{key}"]'
                    continue
                key = int(key) if item == 'args' else key
                data = bound.arguments[key] if item == 'pargs' else bound.arguments[item][key]
                name = f'{key}'
                getattr(self.mmp, meta['comm_type'])(storage='ls', **{name: data})
                arguments[item][key] = bound.arguments[key] if item=='pargs' else bound.arguments[item][key]
                sub_arguments[item][key] = f'mmp.ls["{key}"]'


        # 主进程执行函数

        res = self.func(*arguments['pargs'].values(), *arguments['args'].values(), **arguments['kwargs'])
        if isinstance(self.res_name, str):
            self.mmp.gs[self.res_name] = res
        elif isinstance(self.res_name, tuple):
            for key, value in zip(self.res_name, res):
                self.mmp.gs[key] = value



        # 子进程执行函数
        resp = {'comm_type': 'exec'}
        code = []
        returns = ''
        if isinstance(self.res_name, str):
            self.mmp.gs[self.res_name] = res
            returns = f'mmp.gs["{self.res_name}"]'
        elif isinstance(self.res_name, tuple):
            for key, value in zip(self.res_name, res):
                self.mmp.gs[key] = value
                returns += f'mmp.gs["{key}"], '
        code.append(returns)
        pargs = ', '.join(sub_arguments['pargs'].values()) + ', ' if sub_arguments['pargs'] else ''
        args = ', '.join(sub_arguments['args'].values()) + ', ' if sub_arguments['args'] else ''
        kwargs = ', '.join(f'{key}={value}' for key,value in sub_arguments['kwargs'].items()) + ', ' if sub_arguments['kwargs'] else ''
        code[0] += '= '+self.func.__code__.co_name+'('+pargs+args+kwargs+')'
        code.append('mmp.ls = {}')
        code = '\n'.join(code)
        resp = {'comm_type': 'exec', 'code': code}
        self.mmp.comm.bcast(resp, root=self.mmp.root)


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
    def mode(self):
        return self.mmp.mode
    
    @property
    def root(self):
        return self.mmp.root
    
    @property
    def gs(self):
        return self.mmp.gs
    
    @property
    def ls(self):
        return self.mmp.ls
    
    def log(self, category=None, msg=None):
        return self.mmp.log(category, msg)
    

    def __setitem__(self, key, value):
        if isinstance(key, str):
            kwargs = {key:value}
        if isinstance(key, tuple):
            kwargs = {ikey: ivalue for ikey, ivalue in zip(key, value)}
        self.__call__(**kwargs)
       

    def generate_resp(self, key, value, storage='gs'):
        _type = type(value).__name__
        resp = {
            'comm_type': self.__class__.__name__[3:],
            'name': key,
            'type': _type,
            'dtype': str(value.dtype) if _type == 'ndarray' else '-',
            'shape': value.shape if _type == 'ndarray' else '-',
            'storage': storage,
        }
        return resp
    

    def exit(self):
        self.comm.bcast({'comm_type': 'exit'}, root=self.root)

@MinifyMPI.register_comm_cls
class MPIbcast(MPICommBase):
    def __call__(self, storage='gs', **kwargs: Any) -> None:
        if self.is_main:
            for key, value in kwargs.items():
                resp = self.generate_resp(key, value, storage)
                resp['data'] = value
                self.comm.bcast(resp, root=self.root)
                self.gs[resp['name']] =  value
        else:
            resp = kwargs['resp']
            getattr(self, resp['storage'])[resp['name']] = resp['data']


@MinifyMPI.register_comm_cls
class MPIBcast(MPICommBase):
    def __call__(self, storage='gs', **kwargs: Any) -> None:
        if self.is_main:
            for key, value in kwargs.items():
                resp = self.generate_resp(key, value, storage)
                resp['data'] = value
                self.comm.bcast(resp, root=self.root)
                self.comm.Bcast(value, root=self.root)
                self.gs[resp['name']] =  value
        else:
            resp = kwargs['resp']
            data = np.zeros(resp['shape'], resp['dtype'])
            self.comm.Bcast(data, root=0)
            getattr(self, resp['storage'])[resp['name']] = data
            # self.log('__call__', self.ls)

@MinifyMPI.register_comm_cls
class MPIscatter(MPICommBase):
    def __call__(self, **kwargs: Any) -> None:
        if self.is_main:
            for key, value in kwargs.items():
                if not isinstance(value, (list, np.ndarray)):
                    msg = ('The type of value for Bcast must be one of '
                        f'[`list`, `np.ndarray`], but `{type(value.__name__)}` '
                        f'was given for `{key}`.')
                    self.exit()
                    raise TypeError(msg)
                if len(value) != self.mmp.n_procs:
                    msg = ('The length of data must be equal to the `n_procs`'
                           f'({self.mmp.n_procs}), but the length of "{key}" '
                           f'is {len(value)}.')
                    self.exit()
                    raise ValueError(msg)
                resp = self.generate_resp(key, value)
                self.comm.bcast(resp, root=self.root)
                if self.mode == 'dynamic':
                    self.comm.scatter(value[1:], root=self.root)
                else:
                    self.comm.scatter(value, root=self.root)
                self.gs[resp['name']] =  value[0]


        else:
            resp = kwargs['resp']
            data = self.comm.scatter(None, root=0)
            self.gs[resp['name']] =  data

@MinifyMPI.register_comm_cls
class MPIScatter(MPICommBase):
    def reshape(self, data): 
        '''
        Automatically reorganizes according to the number of main process tasks(`main_task_count`), 
        the number of subprocess(`n_sub_procs`) and the number of sub-process tasks(`sub_task_count`).

        Parameters
        ----------
        data: list
            Data to reorganizes. 

        Returns
        -------
        reorganized_data : list
            reorganized data.

        '''
        if not isinstance(data, (list, )):
            raise TypeError(f'`{type(data).__name__}` does not support automatic allocation.')

        _type = type(data)
        new_data = data[self.mmp.n_tasks_main:]

        new_data = [_type(new_data[i*self.mmp.n_tasks_sub: (i+1)*self.mmp.n_tasks_sub]) for i in range(self.mmp.n_procs - 1)]
        new_data = [data[:self.mmp.n_tasks_main]] + new_data
        return _type(new_data)

    def __call__(self, **kwargs: Any) -> None:
        if self.is_main:
            for key, value in kwargs.items():
                if not isinstance(value, (list, np.ndarray)):
                    msg = ('The type of value for Bcast must be one of '
                        f'[`list`, `np.ndarray`], but `{type(value).__name__}` '
                        f'was given for `{key}`.')
                    self.exit()
                    raise TypeError(msg)
                if len(value) != self.mmp.n_tasks:
                    msg = ('The length of data must be equal to the `n_tasks`'
                           f'({self.mmp.n_tasks}), but the length of "{key}" '
                           f'is {len(value)}.')
                    self.exit()
                    raise ValueError(msg)
                resp = self.generate_resp(key, value)
                self.comm.bcast(resp, root=self.root)
                if resp['type'] == 'ndarray':
                    #TODO - 数据分发改为Send
                    # 数组的数据分发目前动态的是使用Scatter方法，而mpirun是使用Send
                    # 可以考虑统一为Send。数据大小、速度上的差别也可以测试一下。
                    if self.mode == 'dynamic':
                        self.comm.Scatter(value[self.mmp.n_tasks_main:], None, root=self.root)
                        self.gs[resp['name']] =  value[:self.mmp.n_tasks_main]
                    else:
                        self.gs[resp['name']] =  value[:self.mmp.n_tasks_main]
                        starts = np.arange(self.mmp.n_tasks_main, self.mmp.n_tasks, self.mmp.n_tasks_sub)
                        stops = starts + self.mmp.n_tasks_sub
                        slcs = [slice(start, stop) for start, stop in zip(starts, stops)]
                        for i, slc in enumerate(slcs):
                            self.comm.Send(value[slc], dest=i+1)
                else:
                    if self.mode == 'dynamic':
                        value = self.reshape(value)
                        self.comm.scatter(value[1:], root=self.root)
                        self.gs[resp['name']] =  value[0]
                    else:
                        value = self.reshape(value)
                        value = self.comm.scatter(value, root=self.root)
                        self.gs[resp['name']] =  value
        else:
            resp = kwargs['resp']
            if resp['type'] == 'ndarray':
                if self.mode == 'dynamic':
                    data = np.zeros((self.mmp.n_tasks_sub,)+resp['shape'][1:], resp['dtype'])
                    # self.mmp.log('test', self.mmp.n_tasks_sub)

                    self.comm.Scatter(None, data, root=0)
                    self.gs[resp['name']] =  data
                else:
                    data = np.zeros((self.mmp.n_tasks_sub,)+resp['shape'][1:], resp['dtype'])
                    self.comm.Recv(data, source=0)
                    self.gs[resp['name']] =  data
            else:
                data = self.comm.scatter(None, root=0)
                self.gs[resp['name']] =  data

@MinifyMPI.register_comm_cls
class MPIgather(MPICommBase):
    def __getitem__(self, key):
        if isinstance(key, str):
            key = (key, )
        elif isinstance(key, tuple):
            pass
        return self.__call__(*key)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        if self.is_main:
            returns = []
            for key in args:
                resp = {'comm_type': 'gather', 'name': key,}
                self.comm.bcast(resp, root=self.root)
                if self.mode == 'dynamic':
                    returns.append([self.gs[key]])
                    data = self.comm.gather(None, root=self.root)
                    returns[-1].extend(data)
                else:
                    data = self.comm.gather(self.gs[key], root=self.root)
                    returns.append(data)

            return returns[0] if len(returns) == 1 else returns
        else:
            resp = kwargs['resp']
            self.comm.gather(self.gs[resp['name']], root=0)

@MinifyMPI.register_comm_cls
class MPIGather(MPICommBase):
    def recover(self, data):
        '''
        Automatically recover to one list For a list that contains many lists.

        Parameters
        ----------
        data : list
            A list that contains many lists.

        '''
        _types = [type(item) for item in data]
        if _types.count(type(data)) != len(data):
            raise TypeError(f'The element types{[item.__name__ for item in _types]} does '
                            f'not match the data type(`{type(data).__name__}`)')
        return reduce(lambda x, y: x+y, data)


    def __getitem__(self, key):
        if isinstance(key, str):
            key = (key, )
        elif isinstance(key, tuple):
            pass
        return self.__call__(*key)


    def __call__(self, *args: Any, **kwargs: Any) -> None:    
        if self.is_main:

            returns = []
            for key in args:
                resp = {'comm_type': 'Gather', 'name': key,}
                data0 = self.gs[key]
                if isinstance(data0, np.ndarray):
                    shape = (self.mmp.n_tasks, )+data0.shape[1:]
                    resp['type'] = 'ndarray'
                    resp['shape'] = shape

                    #REVIEW - Gather报错
                    # 如果我们通过抛出错误的方法来让用户知道无法进行Gather通信，那么进程间的通信是
                    # 会继续等待进行的，不会自动结束。但是报错之后，如果用户没有用try、catch的写法，
                    # 那么通过py文件来运行的程序，py文件的执行已经停止了，但是进程间通信不会断开，
                    # 这可能会带来一些隐患，例如会卡顿。
                    # 如果是在notebook环境，那么不会有问题。因为notebook还在运行，通信不需要断开。

                    self.comm.bcast(resp, root=self.root)
                    resps = self.comm.gather({'status': 'continue'}, root=self.root)
                    if all([resp['status']=='continue' for resp in resps]):
                        self.comm.bcast({'status': 'continue'}, root=self.root)
                    else:
                        # TODO 目前只检查了type、shape，dtype没有检查
                        self.comm.bcast({'status': 'stop'}, root=self.root)
                        msg = 'The expected value from sub-processes are '+\
                            'ndarray of shape {}. but got:\n'
                        msg = msg.format((self.mmp.n_tasks_sub,)+data0.shape[1:])
                        for i, resp in enumerate(resps):
                            i = i if self.mode == 'mpirun' else i+1
                            if resp['status'] == 'stop':
                                msg += f'sub{i}: {resp["type"]} of shape {resp["shape"]}\n'
                        raise TypeError(msg)

                    data = np.zeros(shape, data0.dtype)
                    data[:self.mmp.n_tasks_main] = data0
                    if self.mode =='dynamic':
                        self.comm.Gather(None, data[self.mmp.n_tasks_main:], root=self.root)
                    else:
                        for i in range(self.mmp.n_procs-1):
                            start = self.mmp.n_tasks_main + self.mmp.n_tasks_sub*i
                            stop = start + self.mmp.n_tasks_sub
                            self.comm.Recv(data[start:stop], source=i+1)
                elif isinstance(data0, list):
                    resp['type'] = 'list'
                    self.comm.bcast(resp, root=self.root)
                    data = self.comm.gather(data0, root=self.root)
                    data = [data0]+data if self.mode == 'dynamic' else data
                    data = self.recover(data)
                returns.append(data)
            return returns[0] if len(returns) == 1 else returns
        else:
            resp = kwargs['resp']
            data = self.gs[resp['name']]
            if resp['type'] == 'ndarray':

                # 检查是不是np数组
                if not isinstance(data, np.ndarray):
                    resp = {'status': 'stop', 'shape': 'None', 'type': type(data).__name__}
                    self.comm.gather(resp, root=0)

                # 检查shape是否符合预期
                elif data.shape[1:] == resp['shape'][1:] \
                    and len(data) == self.mmp.n_tasks_sub:
                    self.comm.gather({'status': 'continue'}, root=0)
                else:
                    resp = {'status': 'stop', 'shape': data.shape, 'type': 'ndarray'}
                    self.comm.gather(resp, root=0)
                

                resp = self.comm.bcast(None, root=0)
                # 如果所有子进程的shape的都是符合预期的，这发送数据
                if resp['status'] == 'continue':
                    if self.mode == 'dynamic':
                        self.comm.Gather(data, None, root=0)
                    else:
                        self.comm.Send(data, dest=0)

                # 否则结束通信
                else:
                    return
            else:
                self.comm.gather(data, root=0)

@MinifyMPI.register_comm_cls
class MPIloc(MPICommBase):
    #NOTE - loc设置数据
    # 这里有些问题需要考虑。
    # 1. 数据检查。我们不会对数据进行检查，你可以把原本是ndarray的数据设置为list，
    #    我们认为这是用户希望的，所以不会报错，但是会弹出提醒。
    # 2. 我们应该支持像pandas.DataFrame 那样的获取和设置逻辑
    # 3. 在获取数值时，我们不会检查子进程是否有指定的变量，而是通过字典的get方法
    #    来获取变量。也就是说，如果没有变量，就会返回`None`
    
    def __getitem__(self, keys):
        if not isinstance(keys, tuple):
            raise IndexError('index error')
        if len(keys) != 2:
            raise IndexError('index error')

        idx = keys[0]
        if not isinstance(idx, (int, tuple, slice, list)):
            raise ValueError('The index must be an integer.')
        
        if isinstance(idx, int):
            idx = (idx - 1, ) if self.mode == 'dynamic' else (idx, )
        elif isinstance(idx, (tuple, list)):
            idx = [i - 1 for i in idx] if self.mode == 'dynamic' else idx
        else:
            start = idx.start if idx.start else 0
            stop = idx.stop if idx.stop else self.mmp.n_procs
            stop = stop if stop > 0 else self.mmp.n_procs + stop
            if self.mode == 'dynamic':
                start -= 1
                stop -= 1
            step = 1 if idx.step == None else idx.step
            idx = list(range(start, stop, step))


        if self.mode == 'dynamic':
            if min(idx) < -1:
                raise IndexError('The index must be greater than or equal to 0.')
            if max(idx)-1 >= self.mmp.n_procs:
                raise IndexError('The process index exceeded the number of processes.')

        else:
            if min(idx) < 0:
                raise IndexError('The index must be greater than or equal to 0.')
            if max(idx) >= self.mmp.n_procs:
                raise IndexError('The process index exceeded the number of processes.')
    

        if isinstance(keys[1], str):
            names = (keys[1], )
        elif isinstance(keys[1], (tuple, list)):
            names = keys[1]
        return self.get(idx, names)


    def get(self, idx=None, names=None, *args, **kwargs):
        if self.is_main:
            returns = []
            for i in idx:
                returns.append([])
                for key in names:
                    resp = {
                        'comm_type': 'loc', 
                        'idx': i, 
                        'name': key,
                        'mode': 'get',
                    }
                    self.comm.bcast(resp, root=self.root)

                    if (self.mode == 'dynamic' and i==-1) or\
                        (self.mode=='mpirun' and i == 0):
                        data = self.gs[key]
                    else:
                        meta = self.comm.recv(source=i)
                        if meta['type'] == 'ndarray':
                            data = np.zeros(meta['shape'], meta['dtype'])
                            self.comm.Recv(data, source=i)
                        else:
                            data = self.comm.recv(source=i)
                    returns[-1].append(data)
            if len(names) == 1:
                returns = [item[0] for item in returns]
            if len(idx) == 1:
                returns = returns[0]
            return returns

        else:
            if self.comm.rank == kwargs['resp']['idx']:
                name = kwargs['resp']['name']
                data = self.gs.get(name)
                if data is None:
                    self.log('warning', f'`{name}` not found in this process, '
                             '`None` will be returned.')
                meta = {
                    'name': name,
                    'type': type(data).__name__,
                    'dtype': data.dtype if isinstance(data, np.ndarray) else '-',
                    'shape': data.shape if isinstance(data, np.ndarray) else '-',
                }
                self.comm.send(meta, dest=0)
                if meta['type'] == 'ndarray':
                    self.comm.Send(data, dest=0)
                else:
                    self.comm.send(data, dest=0)


    def __setitem__(self, keys, values):
        if not isinstance(keys, tuple):
            raise KeyError
        if len(keys) != 2:
            raise KeyError
        
        # 检查键类型和数量
        idx = keys[0]
        names = keys[1]
        if isinstance(idx, int):
            idx = (idx, )
        elif isinstance(idx, slice):
            start = idx.start if idx.start else 0
            stop = idx.stop if idx.stop else self.mmp.n_procs
            stop = stop if stop > 0 else self.mmp.n_procs + stop
            step = 1 if idx.step == None else idx.step
            idx = list(range(start, stop, step))

        elif not isinstance(idx, (tuple, list)):
            raise KeyError
        
        if isinstance(names, str):
            names = names,
        elif not isinstance(names, (tuple, list)):
            raise KeyError
        
        # 检查键和值是否配对
        if len(names) == 1:
            values = values,
        

        elif len(names) > 1:
            if not isinstance(values, (list, tuple)):
                values = [values,]*len(names)
            elif len(names) != len(values):
                raise ValueError
            
        if self.mode == 'dynamic':
            idx = [i-1 for i in idx]    
        self.set(idx, names, values)


    def set(self, idx=None, names=None, values=None, *args, **kwargs):
        if self.is_main:
            for i in idx:
                for name, data in zip(names, values):
                    resp = self.generate_resp(name, data)
                    resp['mode'] = 'set'
                    resp['idx'] = i
                    self.comm.bcast(resp, root=self.root)
                    if (self.mode == 'dynamic' and i == -1) \
                        or (self.mode=='mpirun' and i==0):
                            self.gs[name] = data
                    if resp['type'] == 'ndarray':
                        self.comm.Send(data, dest=i)
                    else:
                        self.comm.send(data, dest=i)

        else:
            resp = kwargs['resp']
            name = resp['name']
            if self.comm.rank == resp['idx']:
                if resp['type'] == 'ndarray':
                    data = np.zeros(resp['shape'], resp['dtype'])
                    self.comm.Recv(data, source=0)
                else:
                    data = self.comm.recv(source=0)

                if name in self.gs and not isinstance(self.gs[name], type(data)):
                    msg = f'{Fore.GREEN}[{self.mmp.proc_id}]{Fore.RESET}' +\
                          f'The type of `{name}` changed ' +\
                          f'from `{type(self.gs[name]).__name__}` to ' +\
                          f'`{type(data).__name__}`'
                    warnings.warn(msg)

                self.gs[name] = data


    def __call__(self, **kwargs: Any) -> None:
        resp = kwargs['resp']
        if resp['mode'] == 'get':
            return self.get(**kwargs)
        elif resp['mode'] == 'set':
            return self.set(**kwargs)


mmp = MinifyMPI()
