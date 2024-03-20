import numpy as np
import sys, os
import unittest
from mpi4py import MPI

size = MPI.COMM_WORLD.size
if size == 1:
    from ..core.notebook import MinifyMPI
else:
    from ..core.base import MinifyMPI

'''
本文件用于代码执行。通过调用mmp.exec，将代码分发给所有进行并执行。
'''

class TestExec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mmp = MinifyMPI(size if size > 1 else 4)
        cls.mmp.start_comm()


    @classmethod
    def tearDownClass(cls):
        cls.mmp.close_comm()


    @property
    def n_procs(self):
        return self.mmp.n_procs


    def test_exec(self):
        self.mmp.bcast['data'] = 1
        self.mmp.exec('mmp.gs["data"] = mmp.gs["data"]*3')
        self.assertListEqual(self.mmp.gather['data'], [3]*self.n_procs)

