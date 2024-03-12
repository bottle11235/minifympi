from .core import MinifyMPI, MPIFunction
from mpi4py import MPI
import inspect
import re
import numpy as np
from textwrap import dedent, indent

# TODO 静态装饰器
# Parallel是动态生成进程来实现的，目前还不支持mpirun的静态的并行装饰器，原理上，静态
# 装饰器也是支持的，也许可以考虑实现它。

class Parallel:
    dct_annotations = {
        's': 'scatter',
        'S': 'Scatter',
        'b': 'bcast',
        'B': 'Bcast',
        'g': 'gather',
        'G': 'Gather',
    }

    def __init__(self) -> None:
        self.mmp = MinifyMPI(mode='dynamic')

    def register_func(self, func):
        # 将函数发送到子进程。
        code = self.mmp.generate_script(func)
        code = code.split('\n')
        new_code = []
        for line in code:
            if line.strip().startswith('@'):
                pattern = '@(.+'+('?)\(' if '(' in line else ')')
                decorator = re.findall(pattern, line)[0]
                if eval(decorator) == self:
                    continue
            new_code.append(line)
        code = '\n'.join(new_code)

        resp = {
            'comm_type': 'exec',
            'code': code,
            'co_name': func.__code__.co_name,
        }
        self.mmp.comm.bcast(resp, root=self.mmp.root)
        mpi_func = MPIFunction(self.mmp, func)
        setattr(self.mmp, func.__code__.co_name, mpi_func)

    def transmit_data(self, func, *args, **kwargs):
        sig = inspect.signature(func)  
        bound = sig.bind(*args, **kwargs)
        annotations = func.__annotations__
        
        # 数据传输
        arguments = {
            'pargs':{}, 
            'args': {str(i): value for i, value in enumerate(bound.arguments['args'])} if 'args' in bound.arguments else {}, 
            'kwargs': bound.arguments['kwargs'] if 'kwargs' in bound.arguments else {}
        }
        arguments['pargs'] = {key: value for key, value in bound.arguments.items() if key not in ['args', 'kwargs']}
        paral_args = {'pargs': {}, 'args':{}, 'kwargs':{}}
        for category, value in arguments.items():
            for name, data in value.items():
                if name in annotations:
                    comm_type = self.dct_annotations[annotations[name]]
                elif isinstance(data, np.ndarray):
                    comm_type = 'Bcast'
                else:
                    comm_type = 'bcast'
                data = arguments[category][name]
                getattr(self.mmp, comm_type)[name] = data
                paral_args[category][name] = self.mmp[name]
        return paral_args


    def get_returns(self, func):
        annts = func.__annotations__
        annts_return = annts['return'].replace(' ', '').split(',') if 'return' in annts else []

        returns = self.mmp.gs['_returns_']
        if not isinstance(returns, tuple):
            returns = returns,

        all_returns = []
        for i, value in enumerate(returns):
            if i < len(annts_return):
                comm_type = self.dct_annotations[annts_return[i]]
            elif isinstance(value, np.ndarray):
                comm_type = 'Gather'
            else:
                comm_type = 'gather'
            name = f"_returns{i}_"
            code = f'mmp.gs["{name}"] = mmp.gs["_returns_"][{i}]'
            resp = {'comm_type': 'exec', 'code': code,}
            self.mmp.comm.bcast(resp, self.mmp.root)
            self.mmp.gs[name] = self.mmp.gs["_returns_"][i]
            data = getattr(self.mmp, comm_type)[name]
            all_returns.append(data)
        return all_returns


    def __call__(self, n_procs=None, n_tasks=None):

        # TODO 自动推导n_task
        # n_procs是必须提供的（对于动态生成进程来说），而n_tasks则可根据Scatter、scatter
        # 数据的长度自动推导，如果没有Scatter、scatter的数据，这n_tasks应当等于n_procs。
        # 当然，手动指定n_tasks也是必须支持的。

        def decorator(func):
            def wrapper(*args, **kwargs):
                # 设置进程和任务数，建立连接
                self.mmp.n_procs = n_procs
                self.mmp.n_tasks = n_tasks
                self.mmp.start_comm({'mmp': self.mmp})

                # 注册函数到子进程
                self.register_func(func)
                # 传输数据到子进程
                paral_args = self.transmit_data(func, *args, **kwargs)
                # 所有进程运行函数
                getattr(self.mmp, func.__code__.co_name)['_returns_'](
                    *paral_args['pargs'].values(),
                    *paral_args['args'].values(),
                    **paral_args['kwargs'],
                )
                # 保存结果到主进程
                all_returns = self.get_returns(func)
                self.mmp.close_comm()
                return all_returns
            return wrapper
        return decorator


parallel = Parallel()


