from numba import jit
from textwrap import indent, dedent
import sys
import ast, astor
import types
import re
import inspect
from pandas.io.parsers.readers import read_csv as rc
import warnings
import black


def find_requires(func, g=None, ignores=None):
    ''' 
    Find dependency libraries and functions for `func`.

    Parameters
    ----------
    func : function
        Any function.
    '''
    ignores = ignores if ignores else ["__builtins__", "__file__"]
    g = g if g else getattr(func, '__globals__', globals())
    requires = find_code_requires(func, g, ignores)
    return {require: g[require] for require in requires if require in g}


def find_code_requires(func, g, ignores=None):
    node = ast.parse(dedent(inspect.getsource(func))).body[0]
    decs = [dec.id if hasattr(dec, 'id') else dec.func.id for dec in node.decorator_list]
    requires = [
        name for name in func.__code__.co_names + tuple(decs)
        if name not in ignores \
        and name != func.__code__.co_name \
        and hasattr(sys.modules['__main__'], name)
    ]
    for item in requires:
        if item in g and isinstance(g[item], types.FunctionType) and g[item].__name__ != '<lambda>':
            for key in find_requires(g[item], ignores=ignores + requires):
                if key not in requires:
                    requires.append(key)
    return requires


def generate_script(func, ignores=None, requires=None):
    '''
    创建依赖函数，包括引入相关的库和被调用的函数等。 值得注意的是，如果函数内部使用了全局变量，
    将会抛出一个存在风险的提示。
    '''
    # 根据函数更新依赖库和依赖函数
    if requires is None:
        requires = dict()
    if isinstance(requires, list):
        requires = {item: globals()[item] for item in requires}
        
    requires.update(find_requires(func, ignores=ignores))
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
        else:
            msg = f'`{key}` is a global variable which may cause unknown errors.'
            warnings.warn(msg, UserWarning)

    code_func.append(inspect.getsource(func))
    code = dedent('\n'.join(code_import + code_func))
    return black.format_str(code, mode=black.Mode())



sys.path.append("/mnt/d/Python/minifympi/")
from minifympi.core.__init__ import a

@jit
def test():
    rc
    a
    test3()

def test2(func):
    print(1)
    return func

@test2
def test3():
    print(3)


print(generate_script(test, ['jit'], ['jit']))