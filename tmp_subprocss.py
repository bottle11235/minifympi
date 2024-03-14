import sys, os
sys.path.append("/mnt/d/Python/minifympi/")
from minifympi.core.notebook import MinifyMPI
mmp = MinifyMPI(4, 10)
mmp.start_comm()