from mpi4py import MPI



class MinifyMPI:
    def __init__(self) -> None:
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank

    
    def i_all_send(self, resp):
        for i in range(self.comm.size):
            req = self.comm.isend(resp, dest=i)
        # req.wait()

    
    @property
    def is_main(self):
        return self.comm.rank == 0

    @property
    def ROOT(self):
        return 0

    def start_comm(self, gs=None):
        self.gs = gs if gs is not None else {'mmp': self}
        self.comm = MPI.COMM_WORLD
        if self.is_main:
            return
        
        while True:
            resp = self.comm.irecv(source=self.ROOT).wait()

            if resp['comm_type'] == 'exit':
                print('exit', resp)
                # #TODO 根据具体环境，检查是否运行Disconnect函数
                # self.comm.Disconnect()
                exit()
            else:
                print(resp)
                pass
                # getattr(self, resp['comm_type']).__comm__(resp=resp)


    
    def close_comm(self):
        resp = {'comm_type': 'exit'}
        self.i_all_send(resp)
        # self.comm.Disconnect()


mmp = MinifyMPI()
mmp.start_comm()

mmp.i_all_send({'comm_type': 'test', 'data': [1, 2]})
mmp.close_comm()






