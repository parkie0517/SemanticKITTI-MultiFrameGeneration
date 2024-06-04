"""
    'grogu.py' is a test file.
    this file is included in the gitignore file
"""

import os
import numpy as np
from numpy.linalg import inv

def load_pc(file_path):
    with open(file_path, 'rb') as file:
        data = np.fromfile(file, dtype=np.float32) 
        point_cloud_data = np.reshape(data, (-1, 4))
        return point_cloud_data





def voxelize(data, pose):

    xyz = data[:, :3]
    sig = data[:, 3:]
    """
    # calibration
    hpoints = np.hstack((data[:, :3], np.ones_like(data[:, :1]))) # create a homogeneous coordinate, so hpoints.shape == (n, 4) 
    new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1) # check
    xyz = new_points[:, :3] # transform back to the (1, 3) shaped coordinate
    """
    output_shape = (256, 256, 32)
    grid_size = np.asarray(output_shape)

    fixed_volume_space = True
    max_volume_space=[51.2, 25.6, 4.4]
    min_volume_space=[0, -25.6, -2]

    max_bound = np.asarray(max_volume_space)
    min_bound = np.asarray(min_volume_space)


    # Filter point cloud
    xyz0 = xyz
    for ci in range(3):
        xyz0[xyz[:, ci] < min_bound[ci], :] = 1000
        xyz0[xyz[:, ci] > max_bound[ci], :] = 1000
    valid_inds = xyz0[:, 0] != 1000
    xyz = xyz[valid_inds, :]
    sig = sig[valid_inds]
  


    # transpose centre coord for x axis
    x_bias = (max_volume_space[0] - min_volume_space[0])/2
    min_bound[0] -= x_bias
    max_bound[0] -= x_bias
    xyz[:, 0] -= x_bias


    # get grid index
    crop_range = max_bound - min_bound
    cur_grid_size = grid_size

    intervals = crop_range / (cur_grid_size - 1)
    
    if (intervals == 0).any(): print("Zero interval!")

    grid_ind = (np.floor((np.clip(xyz, min_bound, max_bound) - min_bound) / intervals)).astype(int)

    # process voxel position
    dim_array = np.ones(len(grid_size) + 1, int)
    dim_array[0] = -1
    voxel_position = np.indices(grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)

    #processed_label = labels  # voxel labels

    #data_tuple = (voxel_position, processed_label)
    data_tuple = (voxel_position)
    print(voxel_position[0][0])
    print(grid_ind[0])
    exit(0)
    data_tuple += (grid_ind)

    print(data.shape)
    exit(0)
    
    return data_tuple


def load_calib(calib_path):
    """
        1. read the calib.txt
        2. find the Tr data
        3. return it
    """
    # Initialize a list to hold the extracted numbers
    tr_numbers = []
    # Open and read the file line by line
    with open(calib_path, 'r') as file:
        for line in file:
            # Check if the line starts with 'Tr:'
            if line.startswith('Tr:'):
                # Split the line into parts after 'Tr:'
                parts = line.split()[1:]  # Skip the 'Tr:' part
                # Convert each part to a float and add to the list
                tr_numbers = [float(num) for num in parts]
                break
    matrix = np.array(tr_numbers).reshape(3, 4)
    matrix = np.vstack([matrix, np.array([0, 0, 0, 1])])

    return matrix


def load_poses(poses_path, Tr, Tr_inv):
    """
    1. pad index with zeros    
    2. read poses.txt file
    """
    poses = []

    with open(poses_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Split the line into floats
            numbers = np.array(list(map(float, line.strip().split())))
            # Reshape into a 3x4 matrix
            pose_matrix = numbers.reshape((3, 4))
            # convert to a 4x4 matrix
            pose_matrix_4x4 = np.vstack([pose_matrix, [0, 0, 0, 1]])
            #pose_matrix_calib = Tr_inv @ (pose_matrix_4x4 @ Tr)
            pose_matrix_calib = np.matmul(Tr_inv, np.matmul(pose_matrix_4x4, Tr))
            poses.append(pose_matrix_calib)

    return poses



dataset = '/mnt/ssd2/jihun/dataset/sequences/00'

# 3. read the calib.txt
calib_location = os.path.join(dataset, "calib.txt")
Tr = load_calib(calib_location)
Tr_inv = inv(Tr)

# 4. read poses.txt file (it's more efficient to read poses.txt just once)
poses_location = os.path.join(dataset, "poses.txt")
poses = load_poses(poses_location, Tr, Tr_inv)


file_path = '/mnt/ssd2/jihun/dataset/sequences/00/velodyne/000000.bin'
pc_data = load_pc(file_path)
grid_ind, return_fea = voxelize(pc_data, poses[0])

voxel_shape = (256, 256, 32)

# Initialize the voxel grid with zeros
voxel_grid = np.zeros(voxel_shape, dtype=np.uint8)

# Set the specified indices to 1
for idx in grid_ind:
    voxel_grid[idx[0], idx[1], idx[2]] = 1


output_dir = '/mnt/ssd2/jihun/dataset_MF/sequences/00/voxels'
i_file_base = '000000'
np.packbits(voxel_grid).tofile(os.path.join(output_dir, f"{i_file_base}.bin"))
