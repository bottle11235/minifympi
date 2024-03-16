from textwrap import dedent
import sys
import ast, astor
import types
import inspect
import warnings
import black
import re
import importlib


def get_decorators(func):    
    node = ast.parse(dedent(inspect.getsource(func))).body[0]
    decs = [astor.to_source(dec).strip() for dec in node.decorator_list]
    decs = {dec: re.findall('^\w+', dec)[0] for dec in decs}
    return decs


def get_requires(func, gs=None, ignores=None):
    ''' 
    Find dependency libraries and functions for `func`.

    Parameters
    ----------
    func : function
        Any function.
    '''
    ignores = ignores if ignores else ["__builtins__", "__file__"]
    gs = {} if gs is None else gs
    if hasattr(func, '__globals__'):
        gs.update(func.__globals__)
    elif hasattr(func, '__wrapped__'):
        gs.update(func.__wrapped__.__globals__)
    requires = get_code_requires(func, gs, ignores)
    return {require: gs[require] for require in requires if require in gs}


def get_code_requires(func, gs, ignores=None):
    decs = get_decorators(func)
    requires = [
        name for name in func.__code__.co_names + tuple(set(decs.values()))
        if name not in ignores \
        and name != func.__code__.co_name \
        and hasattr(sys.modules['__main__'], name)
    ]
    for item in requires:
        if item in gs and isinstance(gs[item], types.FunctionType) and gs[item].__name__ != '<lambda>':
            for key in get_requires(gs[item], ignores=ignores + requires):
                if key not in requires:
                    requires.append(key)
    return requires


def get_source_with_requires(func, gs=None, ignores=None, requires=None, ):
    '''
    创建依赖函数，包括引入相关的库和被调用的函数等。 值得注意的是，如果函数内部使用了全局变量，
    将会抛出一个存在风险的提示。
    '''
    requires = dict() if requires is None else requires
    requires.update(get_requires(func, gs, ignores=ignores))
    code_import, code_func = [], []
    
    for key, value in requires.items():
        if isinstance(value, types.ModuleType):
            item = f'import {value.__name__}'
            item += f' as {key}' if key != value.__name__ else ''
            code_import.append(item)
        elif isinstance(value, types.FunctionType):
            if value.__module__ == '__main__':
                code_func.append(inspect.getsource(value))
            else:
                item = f'from {value.__module__} import {value.__name__}' 
                item += f' as {key}' if key != value.__name__ else ''
                code_import.append(item)
        elif hasattr(value, '__module__'):
            module = importlib.import_module(value.__module__)
            if hasattr(module, key):
                code_import.append(f'from {value.__module__} import {key}')
            else:
                msg = f'`{key}` is a global variable which may cause unknown errors.'
                warnings.warn(msg, UserWarning)
        else:
            msg = f'`{key}` is a global variable which may cause unknown errors.'
            warnings.warn(msg, UserWarning)

    code_func.append(inspect.getsource(func))
    code = dedent('\n'.join(code_import + code_func))
    return black.format_str(code, mode=black.Mode())


__all__ = ['get_source_with_requires', 'get_decorators']
