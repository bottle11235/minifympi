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


sys.path.append("/mnt/d/Python/minifympi/")
from minifympi.utils.code import get_source_with_requires

@jit
def test():
    rc
    test3()

def test2(func):
    print(1)
    return func

@test2
def test3():
    print(3)


print(get_source_with_requires(test))