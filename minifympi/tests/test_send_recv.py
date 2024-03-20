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
本文件用于测试数据发送和接收。
'''

class TestSetItem(unittest.TestCase):
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


    def test_bcast_gather(self, ):
        self.mmp.bcast['data'] = 1
        data = self.mmp.gather['data']
        self.assertListEqual(data, [1]*self.n_procs)


    def test_Bcast_Gather(self):
        data = np.array([1, 2])
        self.mmp.Bcast['data'] = data
        recvdata = self.mmp.Gather['data']
        np.testing.assert_array_equal(recvdata, np.hstack([data]*self.n_procs))


    def test_scatter_gather(self):
        data = list(range(self.n_procs))
        self.mmp.scatter['data'] = data
        self.assertListEqual(self.mmp.gather['data'], data)


    def test_Scatter_Gather(self):
        data = np.arange(self.n_procs)
        self.mmp.Scatter['data'] = data
        np.testing.assert_array_equal(self.mmp.Gather['data'], data)


    def test_Scatterv_Gatherv(self):
        data = np.arange(self.n_procs*2+2)
        self.mmp.Scatterv['data'] = data
        np.testing.assert_array_equal(self.mmp.Gatherv['data'], data)


    def test_bcast_gather_func(self):
        self.mmp.bcast(storage='ls', data=1)
        data = self.mmp.gather('data', storage='ls')
        self.assertListEqual(data, [1]*self.n_procs)


    def test_Bcast_Gather_func(self):
        data = np.array([1, 2])
        self.mmp.Bcast(storage='ls', data=data)
        recv_data = self.mmp.Gather('data', storage='ls')
        np.testing.assert_array_equal(recv_data, np.hstack([data]*self.n_procs))


    def test_send_recv(self):
        self.mmp.send[2, 'a'] = 5
        self.assertEqual(5, self.mmp.recv[2, 'a'])

    
    def test_Send_Recv(self):
        data = np.arange(3)
        self.mmp.Send[1, 'a'] = data
        np.testing.assert_array_equal(data, self.mmp.Recv[1, 'a'])


