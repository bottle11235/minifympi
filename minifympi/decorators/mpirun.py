from .base import Parallel
from ..core.mpirun import MinifyMPI

parallel = Parallel(MinifyMPI)
parallel.mmp.start_comm()
