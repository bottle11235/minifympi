import unittest
import sys
import numpy as np
sys.path.append("/mnt/d/Python/minifympi/")

from minifympi.core import MinifyMPI
import argparse


# 测试命令
# 1. 以dynamic方式来进行测试。
#    python test_mmp.py
# 2. 目前无法以mpirun的方式来运行测试
#    


mmp = MinifyMPI(3, 10, mode='dynamic')


class Case(unittest.TestCase):
    def setUp(self) -> None:
        mmp.start_comm()
        
    # @unittest.skip
    def test_bcast(self):
        mmp.bcast['data'] = 3
        self.assertEqual(mmp.gs['data'], 3)
        self.assertEqual(mmp.gather['data'], [3]*mmp.n_procs)

        data = np.arange(3)
        mmp.bcast['data'] = data
        self.assertEqual(all(mmp.gs['data']==data), True)
        for item in mmp.gather['data']:
            np.testing.assert_array_equal(item, data)


    def test_Bcast(self):
        data = np.arange(mmp.n_tasks*3).reshape(mmp.n_tasks, 3)
        mmp.Bcast['data'] = data
        np.testing.assert_array_equal(mmp.gs['data'], data)
        np.testing.assert_array_equal(mmp.gather['data'][-1], data)

    
    def test_scatter(self):
        data = list([[1, 2]]*mmp.n_procs)
        mmp.scatter['data'] = data
        self.assertEqual(mmp.gs['data'], data[0])
        self.assertEqual((mmp.gather['data']), data)


    def test_Scatter_Gather(self):
        # 测试数组的Scatter功能，通Gather回收数组
        data = np.arange(mmp.n_tasks*3).reshape(mmp.n_tasks, 3)
        mmp.Scatter['data'] = data
        np.testing.assert_array_equal(mmp.gs['data'], data[:mmp.n_tasks_main])
        np.testing.assert_array_equal(mmp.Gather['data'], data)

        # 测试列表的Scatter功能，通Gather回收列表
        data = list(range(mmp.n_tasks))
        mmp.Scatter['data'] = data
        self.assertEqual(mmp.gs['data'], data[:mmp.n_tasks_main])
        self.assertEqual(mmp.Gather['data'], data)


    def test_loc_get_data(self):
        # 准备数据
        a = list(range(mmp.n_tasks))
        b = np.arange(2*mmp.n_tasks).reshape(mmp.n_tasks, -1)
        mmp.Scatter['a'] = a
        mmp.Scatter['b'] = b

        # 测试loc获取特定进程的数据
        res1 = mmp.loc[0, 'a']
        res2 = mmp.loc[[0, 2], 'a']
        res3 = mmp.loc[2, 'b']
        res4 = mmp.loc[::2, 'a']
        
        # 检测结果
        self.assertListEqual(res1, [0, 1, 2, 3,])
        self.assertListEqual(res2, [[0, 1, 2, 3,], [7, 8, 9]])
        np.testing.assert_array_equal(res3, np.array([[14, 15], [16, 17], [18, 19]]))
        self.assertListEqual(res4, [[0, 1, 2, 3,], [7, 8, 9]])


    def test_loc_set_data(self):
        # 测试设置数据
        mmp.loc[1, 'c'] = 5
        self.assertEqual(mmp.loc[1, 'c'], 5)

        # 测试设置列表数据
        mmp.loc[1, 'c'] = [1, 2]
        self.assertEqual(mmp.loc[1, 'c'], [1, 2])

        # 测试列表index
        mmp.loc[[1, 2], 'c'] = 5
        for i in range(1, 3):
            self.assertEqual(mmp.loc[i, 'c'], 5)

        # 一次设置多个变量
        mmp.loc[[1, 2], ['c', 'd']] = 5, [4, 5]
        self.assertListEqual(mmp.loc[1, ['c', 'd']], [5, [4, 5]])
        self.assertListEqual(mmp.loc[2, ['c', 'd']], [5, [4, 5]])

        # 切片设置变量
        mmp.loc[:3:2, 'c'] = 11
        self.assertListEqual(mmp.loc[:, 'c'], [11, 5, 11])

        # 设置np数组
        mmp.loc[1, 'd'] = np.array([1, 2])
        np.testing.assert_array_equal(mmp.loc[1, 'd'], np.array([1, 2]))

        # 分发ndarray数组，然后修改其中一个进程的数据为list，
        # 然后Gather获取数据，此时应当报错
        with self.assertRaises(TypeError):
            a = np.arange(mmp.n_tasks*2).reshape(mmp.n_tasks, -1)
            mmp.Scatter['a'] = a
            mmp.loc[2, 'a'] = [1, 2]
            mmp.Gather['a']


        # 分发list数据，然后修改其中一个进程的数据为ndarray数组，
        # 然后Gather获取数据，此时应当报错
        with self.assertRaises(TypeError):
            mmp.Scatter['a'] = [i for i in range(mmp.n_tasks)]
            mmp.loc[2, 'a'] = np.array([1, 2])
            mmp.Gather['a']
        

    def test_parallel(self):
        ''' 测试并行函数装饰器 '''
        # prepare data
        a = np.arange(mmp.n_tasks*3).reshape(mmp.n_tasks, 3)
        b = list(range(mmp.n_tasks))
        mmp.Scatter['a'] = a
        mmp.Scatter['b'] = b
        d = np.array([1, 3])
        e = [1, 2]

        # decorate the function
        @mmp.parallel
        def test(a, b, c, *args, **kwargs):
            '''test'''
            return a, b, c, args[0], kwargs['e']
        
        # run parallel function
        mmp.test['res1', 'res2', 'res3', 'res4', 'res5'](mmp['a'], mmp['b'], 4, d, e=e)
        res1, res2 = mmp.Gather['res1', 'res2']
        res3, res4, res5 = mmp.gather['res3', 'res4', 'res5']

        # Check whether the results are correct.
        np.testing.assert_array_equal(res1, a)
        self.assertEqual(res2, b)
        self.assertEqual(res3, [4]*mmp.n_procs)
        for item in res4:
            np.testing.assert_array_equal(item, d)
        self.assertListEqual(res5, [e]*mmp.n_procs)


    def test_del(self):
        '''测试全进程删除变量'''
        a = list(range(mmp.n_tasks))
        b = np.arange(2*mmp.n_tasks).reshape(mmp.n_tasks, -1)

        mmp.Scatter['a'] = a
        mmp.Scatter['b'] = b

        del mmp['a'], mmp['b']

        # 检查进程中是否有`a`,`b`变量
        self.assertEqual(mmp.loc[2, 'a'], None)
        with self.assertRaises(KeyError):
            mmp.Gather['a']


    def test_exec(self):
        pass
        # code = '''
        # 1/0
        # '''
        # print(code)
        # resp = {
        #     'comm_type': 'exec',
        #     'code': code,
        # }
        # mmp.comm.bcast(resp)


    def tearDown(self) -> None:
        # print('finish')
        try:
            mmp.close_comm()
        except Exception as e:
            print(e)
            exit()
        pass
    
    


if __name__ == '__main__':
    unittest.main()

    # suite = unittest.TestSuite()
    # tests = [
    #     Case('test_exec'),

    # ]
    # suite.addTests(tests)
    # runner = unittest.TextTestRunner(verbosity=2)
    # runner.run(suite)

