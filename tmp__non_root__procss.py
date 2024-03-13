import sys, os
sys.path.append("/mnt/d/Python/minifympi/")
from minifympi.dev import MinifyMPINB
mmp = MinifyMPINB(4, 10)
mmp.start_comm()