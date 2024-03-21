from .base import MinifyMPIBase
from mpi4py import MPI
import sys, os

class MinifyMPI(MinifyMPIBase):    
    def __init__(self, n_procs=None) -> None:
        self._n_procs = n_procs
        super().__init__()

    @property
    def n_procs(self):
        return self._n_procs
    
    @n_procs.setter
    def n_procs(self, value):
        if not isinstance(value, int):
            raise TypeError(f'`n_procs` can only be an integer, but `{type(value).__name__}` was given')
        self._n_procs = value

    @property
    def is_main(self):
        return self.comm.Get_parent() == MPI.COMM_NULL

    @property
    def ROOT(self):
        return MPI.ROOT if self.is_main else 0
    
    @property
    def is_root(self):
        if self.is_main:
            return False
        else:
            return self.comm_world.rank == 0

    def start_comm(self, gs=None):
        self.gs = gs if gs is not None else {'mmp': self}
        # self.assign_tasks()

        #STUB - 删除临时库
        # 开发完成就去掉临时库的部分
        code = [
            'import sys, os',
            'sys.path.append(__file__.split("minifympi/core")[0])',
            'from minifympi.core.notebook import MinifyMPI',
            f'mmp = MinifyMPI({self.n_procs})',
            'mmp.start_comm()',
        ]
        
        code = '\n'.join(code)
        fpath = f'{os.path.dirname(__file__)}/tmp_subprocss.py'
        with open(fpath, 'w') as f:
            f.write(code)

        if MPI.Comm.Get_parent() == MPI.COMM_NULL:
            self.comm = MPI.COMM_SELF.Spawn(sys.executable,
                args=[fpath], maxprocs=self.n_procs)
            return
        
        self.comm = MPI.Comm.Get_parent()
        while True:
            resp = self.comm.recv(source=self.ROOT)
            # self.log('recv', resp)
            # self.log('gs', self.gs)
            if resp['comm_type'] == 'exit':
                # self.log('gs', self.gs)
                self.comm.Disconnect()
                exit()
            else:
                getattr(self, resp['comm_type']).__comm__(resp=resp)
