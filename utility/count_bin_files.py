import os

def count_bin_files(directory):
    bin_file_count = 0

    # List of subfolders to iterate over
    subfolders = [f"{i:02d}" for i in range(11)]

    for subfolder in subfolders:
        
        subdir_path = os.path.join(directory, subfolder, 'voxels')
        
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith('.bin'):
                    bin_file_count += 1

    return bin_file_count

# Define the path to your main directory
main_directory = '/mnt/ssd2/jihun/dataset_MF/sequences'

# Call the function and print the result
bin_files = count_bin_files(main_directory)
print(f"Total number of .bin files: {bin_files}")