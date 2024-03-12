import sys, os
sys.path.append("/mnt/d/Python/minifympi/")
from minifympi.core import MinifyMPI
mmp = MinifyMPI(4, 8)
mmp.start_comm(globals())