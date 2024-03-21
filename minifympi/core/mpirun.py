from mpi4py import MPI
from .base import MinifyMPIBase

class MinifyMPI(MinifyMPIBase):    
    def start_comm(self, gs=None):
        self.gs = gs if gs is not None else {'mmp': self}
        self.comm = MPI.COMM_WORLD
        if self.is_main:
            return
        
        while True:
            resp = self.comm.recv(source=self.ROOT)

            if resp['comm_type'] == 'exit':
                exit()
            else:
                getattr(self, resp['comm_type']).__comm__(resp=resp)
