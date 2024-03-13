#%%
import numpy as np
import sys
sys.path.append('/mnt/d/Python/minifympi')
from minifympi.dev import MinifyMPIBase


# %%
mmp = MinifyMPIBase(n_procs=4, n_tasks=22)
# %%
mmp.tasks_count

# %%
n_tasks = 0
n_procs = 4
tasks_count = np.arange(n_procs) 
tasks_count[:] = n_tasks // n_procs
tasks_count[:n_tasks%n_procs] += 1
print(tasks_count)


# %%
np.array_split(np.arange(0), n_procs)
# np.split(np.arange(10), 5, )

# %%
