from mpi4py import MPI
import numpy as np
import sys
import time

comm = MPI.COMM_SELF.Spawn(sys.executable,
                           args=['sub.py'],
                           maxprocs=4)

