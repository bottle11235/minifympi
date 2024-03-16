# from .notebook import MinifyMPI
from mpi4py import MPI
import inspect
# import re
import numpy as np
from ..utils.code import get_source_with_requires, get_decorators
from functools import wraps, update_wrapper, reduce
from itertools import chain
from textwrap import dedent

#TODO
# - 检查返回元组与函数注解的长度
#   当它们不一致时，应当报错
#   def test(a, b)->'g,G':
#       return a
# - 检查函数返回标注是否合理
#   有时候，标注为'G'，但是数组的shape不一致，或者根本不是np.ndarray，就应当报错 


class Parallel:
    dct_annotations = {
        's': 'scatter',
        'S': 'Scatter',
        'b': 'bcast',
        'B': 'Bcast',
        'g': 'gather',
        'G': 'Gather',
        'Sv': 'Scatterv',
        'Gv': 'Gatherv'
    }
    dct_annotations.update({value: value for value in dct_annotations.values()})
    
    def __init__(self, MinifyMPI) -> None:
        self.mmp = MinifyMPI()
        pass

    def register_func(self, func, gs=None, ignores=None, requires=None):
        gs = {} if gs is None else gs
        if hasattr(func, '__globals__'):
            gs.update(getattr(func, '__globals__'))
        elif hasattr(func, '__wrapped__'):
            gs.update(func.__wrapped__.__globals__)

        alias = [key for key, value in gs.items() if value is self]
        ignores = [] if ignores is None else ignores
        ignores.extend(alias)
        code = get_source_with_requires(func, gs, ignores, requires)
        decs = get_decorators(func)
        for dec_code, dec in decs.items():
            if dec in alias:
                code = code.replace('@'+dec_code, '#@'+dec_code)
        self.mmp.exec(code)
        return func


    def recv_data(self, func):
        annts = func.__annotations__
        annts_return = annts['return'].replace(' ', '').split(',') if 'return' in annts else []
        annts_return = {f'_return{i}_': self.dct_annotations[value] for i, value in enumerate(annts_return) if value}
        code = '''
        import numpy as np

        meta = {}
        returns = mmp.ls['_returns_'] 
        returns = returns if isinstance(returns, tuple) else (returns,)
        for i, value in enumerate(returns):
            key = f'_return{i}_'
            mmp.ls[key] = value
            if isinstance(value, np.ndarray):
                if value.shape[0] % mmp.n_procs == 0:
                    meta[key] = mmp.Gather.generate_resp(key, value, 'ls')
                else:
                    meta[key] = mmp.Gatherv.generate_resp(key, value, 'ls')
            else:
                meta[key] = mmp.gather.generate_resp(key, value, 'ls')
        mmp.ls['_meta_'] = meta
        # mmp.log("ls", mmp.ls)
        '''
        self.mmp.exec(dedent(code))
        if '_meta_' not in self.mmp.ls:
            self.mmp.ls['_meta_'] = None 
        _meta_ = self.mmp.gather('_meta_', storage='ls')
        returns = []
        for key, meta in _meta_[0].items():
            if meta['name'] not in self.mmp.ls:
                self.mmp.ls[meta['name']] = None
            comm_type = annts_return.get(key, meta['comm_type'])
            returns.append(getattr(self.mmp, comm_type)(meta['name'], storage='ls'))
        return returns[0] if len(returns) == 1 else tuple(returns)

    def send_data(self, func, *args, **kwargs):
        sig = inspect.signature(func)  
        _bound = sig.bind(*args, **kwargs)
        bound = {}
        bound['pargs'] = _bound.arguments
        bound['args'] = {f'{i}': item for i, item in enumerate(_bound.arguments.pop('args', {}))}
        bound['kwargs'] = _bound.arguments.pop('kwargs', {})
        annotations = func.__annotations__
        
        arguments = {}
        for arg_type, dct in bound.items():
            arguments[arg_type] = []
            for key, value in dct.items():
                prefix = f'{key}=' if arg_type=='kwargs' else ''
                if key in annotations:
                    getattr(self.mmp, self.dct_annotations[annotations[key]])(**{key: value}, storage='ls')
                elif isinstance(value, np.ndarray):
                    self.mmp.Bcast(storage='ls', **{key: value})
                else:
                    self.mmp.bcast(storage='ls', **{key: value})
                arguments[arg_type].append(f'{prefix}mmp.ls["{key}"]')
        return arguments
    
    
    def __call__(self, n_procs=None, gs=None, ignores=None, requires=None):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                self.mmp.n_procs = n_procs
                self.mmp.start_comm()
                self.register_func(func, gs, ignores, requires)
                argument_code = self.send_data(func, *args, **kwargs)
                code_args = ', '.join((chain(*argument_code.values())))
                code = f'mmp.ls["_returns_"] = {func.__code__.co_name}({code_args})'
                self.mmp.exec(code)
                returns = self.recv_data(func)
                self.mmp.close_comm()
                return returns
            return wrapper
        return decorator
        




