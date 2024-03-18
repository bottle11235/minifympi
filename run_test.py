from minifympi.tests.test_send_recv import *
import unittest

from mpi4py import MPI


size = MPI.COMM_WORLD.size
if size == 1:
    from minifympi.core.notebook import MinifyMPI
else:
    from minifympi.core.base import MinifyMPI

#SECTION - non-blocking
# mmp = MinifyMPI(size if size > 1 else 4)
# mmp.start_comm()

# mmp.iloc[1, ['a', 'b']] = [1, 2]
# mmp.iloc[2, 'a'] = np.arange(3)
# mmp.iloc[2, 'a'] = 1
# mmp.exec('mmp.log("gs", mmp.gs)')

# mmp.isend[2, 'a'] = 5
# a = mmp.irecv[2, 'a']
# mmp.log('a', a)
# mmp.exec('mmp.log("gs", mmp.gs)')
# mmp.close_comm()
#!SECTION

#SECTION - send
# mmp = MinifyMPI(size if size > 1 else 4)
# mmp.start_comm()
# mmp.bcast['a'] = 1

# mmp.close_comm()
#!SECTION


if __name__ == '__main__':
    unittest.main()
