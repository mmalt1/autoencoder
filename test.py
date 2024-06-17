import os
import numpy as np

dir = "./spec_arrays"

for array in os.scandir(dir):
    print(array)
    array = np.load(array)
    print(array.size)
    print(array.shape)


 