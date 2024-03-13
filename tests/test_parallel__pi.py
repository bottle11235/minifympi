import time
import numpy as np
import pandas as pd
import sys
sys.path.append("/mnt/d/Python/minifympi/")
from minifympi.wrappers import parallel

def generate_data(num_samples):
    data = np.random.rand(num_samples, 2)
    return data

def monte_carlo_pi(data):
    points_inside_circle = 0
    
    for x, y in data:
        distance = x**2 + y**2  
        
        if distance <= 1:
            points_inside_circle += 1
    
    pi_estimate = 4 * points_inside_circle 
    print(pi_estimate)
    return pi_estimate

@parallel(4, 2000)
def run(
    data1:"S", *args, **kwargs
)-> 'g':
    local_pi_estimate = monte_carlo_pi(data1)
    return local_pi_estimate,

if __name__ == "__main__":
    data = generate_data(2000)
    time_start =  time.perf_counter()
    
    pi_estimate_list = run(data)
    print(pi_estimate_list)
    
    time_end = time.perf_counter()
    print(f"Time elapsed: {time_end - time_start} seconds")

