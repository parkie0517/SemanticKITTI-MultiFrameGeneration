"""
    This file is used to generate a voxel data filled with 1s.
"""


import numpy as np
from numpy.linalg import inv
import argparse
import os
import time
import shutil



# Function to read and reshape binary data
def load_binary_data(file_path):
    with open(file_path, 'rb') as file:
        data = np.fromfile(file, dtype=np.uint8)  # Read data as 8-bit unsigned integers
        bits = np.unpackbits(data)  # Convert bytes to bits
        return bits.reshape((256, 256, 32))  # Reshape to 3D array




if __name__ == '__main__':

    file_path = '/mnt/ssd2/jihun/dataset/sequences/00/voxels/000000.bin'
    bin = load_binary_data(file_path)
    print(bin.shape)
    for i in range(256):
        for j in range(256):
            for k in range(32):
                bin[i, j, k] = 1
    print(bin.shape)

    # Save fused scan
    np.packbits(bin).tofile("./ones.bin")


