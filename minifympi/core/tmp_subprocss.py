import sys, os
sys.path.append(__file__.split("minifympi/core")[0])
from minifympi.core.notebook import MinifyMPI
mmp = MinifyMPI(4)
mmp.start_comm()