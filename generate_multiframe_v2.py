"""
    This file 'generate_multiframe_v2.py' is used to create multiframe semantic KITTI dataset.
    Created by Heejun Park :)
"""


import numpy as np
from numpy.linalg import inv
import argparse
import os
import time
import shutil


def count_files(directory):
    """
        Returns the number of files in the specified directory
    """
    full_path = os.path.abspath(directory) # get the full path name
    items = os.listdir(full_path) # get the list of files in the path
    # Filter out directories, count only files
    file_count = sum(os.path.isfile(os.path.join(full_path, item)) for item in items)
    
    return file_count


def get_pc(file_base, dataset_path):
    """
    Description
    This Function is used to read the point cloud 
    ---------- ---------- ---------- ---------- ---------- 
    Input
    file_base: the name of the data we want to read (eg. 000000)
    dataset_path: location of the original dataset
    ---------- ---------- ---------- ---------- ---------- 
    Below are the main steps
    1. Define point cloud data's location
    2. Read the file
    3. Return the read data
    ---------- ---------- ---------- ---------- ---------- 
    Output
    i_pc
    """


    # Function used to read and reshape point cloud data
    def load_pc_data(file_path):
        with open(file_path, 'rb') as file:
            data = np.fromfile(file, dtype=np.float32) # Point cloud data are stored as 4 byte float type 
            point_cloud_data = np.reshape(data, (-1, 4))
            return point_cloud_data


    # build file path
    pc_path = os.path.join(dataset_path, "velodyne", f"{file_base}.bin")

    # 2. read the data
    pc_data = load_pc_data(pc_path)

    # 3. return the data
    return pc_data


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


def fuse_multiscan(i_pc, j_pc, i_pose, j_pose):
    """
    ---------- ---------- ---------- ---------- ---------- 
    Description
    This function is used to add j into i
    ---------- ---------- ---------- ---------- ---------- 
    Input
    i_pc: i's point cloud data
    j_pc: j's point cloud data
    i_pose: i's pose information
    j_pose: j's pose information
    ---------- ---------- ---------- ---------- ---------- 
    Below are the main steps
    1. Fuse multiscan using the pose information
    ---------- ---------- ---------- ---------- ---------- 
    Output
    fused i + j in point cloud format
    ---------- ---------- ---------- ---------- ---------- 
    """
    
    """ 1. Fuse using the pose information """
    # 1.1. Convert j_pc into World Coodinate System 
    hpoints = np.hstack((j_pc[:, :3], np.ones_like(j_pc[:, :1]))) # create a homogeneous coordinate, so hpoints.shape == (n, 4) 
    new_points = np.sum(np.expand_dims(hpoints, 2) * j_pose.T, axis=1) # check
    new_points = new_points[:, :3] # transform back to the (1, 3) shaped coordinate
    
    # 1.2. Convert j_pc into i_pc Coordinate System
    new_coords = new_points - i_pose[:3, 3] # apply translation difference
    new_coords = np.sum(np.expand_dims(new_coords, 2) * i_pose[:3, :3], axis=1) # apply rotation difference
    new_coords = np.hstack((new_coords, j_pc[:, 3:])) # add the 4th column back to the transformed coordinates

    # 1.3. Fuse i_pc and transformed j_pc
    i_pc = np.concatenate((i_pc, new_coords), 0)


    return i_pc



def voxelize(data):
    xyz = data[:, :3]
    sig = data[:, 3:]

    output_shape = (256, 256, 32)
    grid_size = np.asarray(output_shape)

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

    # voxelize
    # Initialize the voxel grid with zeros
    voxel_grid = np.zeros(output_shape, dtype=np.uint8)

    # Set the specified indices to 1
    for idx in grid_ind:
        voxel_grid[idx[0], idx[1], idx[2]] = 1


    return voxel_grid


if __name__ == '__main__':
    start_time = time.time()
    print('######################################################################')
    print('########## Settings Before Generatating MultiFrame Dataset ###########')
    print('######################################################################')

    # 1. argument settings
    parser = argparse.ArgumentParser(
        description='code for generating multiframe semantic-KITTI dataset for semantic scene completion task'
    )

    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='should be like "..../dataset/sequences/00',
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='type in the output directory',
    )

    parser.add_argument(
        '--number', '-n',
        default='4',
        type=int,
        required=False,
        help='number of frames used to create the multiframe data',
    )

    parser.add_argument(
        '--increment', '-i',
        default=5,
        type=int,
        required=False,
        help='increment size. default is 5',
    )

    args = parser.parse_args() # returns the arguments provided by the user or the default
    dataset = args.dataset
    output = args.output
    n = args.number
    increment = args.increment
    

    # this should be automatically done
    pc_location = os.path.join(dataset, "velodyne/")

    # 2. output directory settings
    output_dir = os.path.join(output, "voxels")
    if os.path.exists(output_dir):
        print("output directory already exists")
    else:
        os.makedirs(output_dir)
        print("output directory does not exist")
        print(f'{output_dir} path has been created')

    

    number_input_files = count_files(pc_location)
    number_distinct_input_files = int(number_input_files/increment) + 1  # we want to use 0, 5, 10, ....   
    sequence_length = (number_distinct_input_files-1)*increment

    # Variblaes needed for printing the progress
    number_output_files = number_distinct_input_files - (n-1)
    progress_interval_percent = 10 # print for every 10 percent
    progress_interval = number_output_files//progress_interval_percent 


    # Copy 'poses.txt', 'calib.txt', 'times.txt' to the destination folder  
    shutil.copy(os.path.join(dataset, "poses.txt"), output)
    shutil.copy(os.path.join(dataset, "calib.txt"), output)
    shutil.copy(os.path.join(dataset, "times.txt"), output)


    print(f'Location of dataset: {dataset}')
    print(f'Location of output directory: {output}')
    print(f'multiframe length: {n}')
    print(f'increment size: {increment}')
    print(f'files from 0 ~ {sequence_length} will be used')
    print(f'total number of output files: {number_output_files}')
    

    # 3. read the calib.txt
    calib_location = os.path.join(dataset, "calib.txt")
    Tr = load_calib(calib_location)
    Tr_inv = inv(Tr)


    # 4. read poses.txt file (it's more efficient to read poses.txt just once)
    poses_location = os.path.join(dataset, "poses.txt")
    poses = load_poses(poses_location, Tr, Tr_inv)


    print('######################################################################')
    print('############### Begin Generatating MultiFrame Dataset ################')
    print('######################################################################')

    # Used for printing out the passed time during execution
    start_time = time.time()

    # algorithm for creating the multi-frame semantic KITTI dataset
    for i in range(0, sequence_length - increment * (n-2), increment):


        # Read necessary files
        i_file_base = f"{i:06d}" # Convert i's data type from INT to STR and pad 0 at the front
        i_pc = get_pc(i_file_base, dataset) # read i-th point cloud data
        i_pose = poses[i] # read i-th pose


        # Copy label, invalid, occluded files into the destination folder
        shutil.copy(os.path.join(dataset, "voxels", f"{i_file_base}.label"), output_dir)
        shutil.copy(os.path.join(dataset, "voxels", f"{i_file_base}.invalid"), output_dir)
        shutil.copy(os.path.join(dataset, "voxels", f"{i_file_base}.occluded"), output_dir)
    

        # Repeat to fuse differenet lidar scans
        for j in range(i + increment, i + increment * n, increment):

            # Read necessary files
            j_file_base = f"{j:06d}" # Convert j's data type from INT to STR and pad 0 at the front
            j_pc = get_pc(j_file_base, dataset) # read j-th point cloud data
            j_pose = poses[j] # read j-th pose
 
            i_pc = fuse_multiscan(i_pc, j_pc, i_pose, j_pose)
        

        # Voxelize fused multiscan
        voxel_grid = voxelize(i_pc)

        # Save fused scan
        np.packbits(voxel_grid).tofile(os.path.join(output_dir, f"{i_file_base}.bin"))
        if (i == 20):
            exit(0)

        # Print progress & time
        if (i!= 0) and (i != progress_interval*increment*10) and (i % (progress_interval*increment) == 0.0):
            end_time = time.time()
            elapsed_time = end_time - start_time
            minutes_passed = int(elapsed_time / 60)
            seconds_passed = int(elapsed_time % 60)
            unge = sequence_length - increment * (n-1)
            print(f'Progress: {i/unge*100.0:.2f}%, Time Passes: {minutes_passed}:{seconds_passed:02d}')
    
    # Print final progress and time
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes_passed = int(elapsed_time / 60)
    seconds_passed = int(elapsed_time % 60)
    unge = sequence_length - increment * (n-1)
    print(f'Progress: 100.00%, Time Passes: {minutes_passed}:{seconds_passed:02d}')
    print("Complete :D")