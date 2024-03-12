#%%
import inspect
import sys
import types
import numpy as np

# %%
def find_requires(func, ignores=None):
    ''' 
    Find dependency libraries and functions for `func`.

    Parameters
    ----------
    func : function
        Any function.
    '''
    ignores = ignores if ignores else ["__builtins__", "__file__"]
    g = dict(func.__globals__)
    print('g', g)
    requires = find_requires_code(func, g, ignores)
    print(requires)
    return {require: g[require] for require in requires if require in g}


def find_requires_code(func, g, ignores=None):
    requires = [
        name for name in func.__code__.co_names
        if name not in ignores \
        and name != func.__code__.co_name \
        # and hasattr(sys.modules['__main__'], name)
    ]
    for item in requires:
        if item in g and isinstance(g[item], types.FunctionType):
            for key in find_requires(g[item], ignores + requires):
                if key not in requires:
                    requires.append(key)
    print(requires)
    return requires

# %%
def decorator(a=2):
    def wrapper(func):
        return func
    return wrapper

@decorator()
def test(a, b):
    np.sin(a)
    return a + b


# %%
find_requires(test)

# %%
