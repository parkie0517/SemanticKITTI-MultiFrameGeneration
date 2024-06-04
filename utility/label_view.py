import numpy as np


def load_pc(file_path):
    with open(file_path, 'rb') as file:
        data = np.fromfile(file, dtype=np.float32) 
        point_cloud_data = np.reshape(data, (-1, 4))
        return point_cloud_data

def load_label(file_path):
    with open(file_path, 'rb') as file:
        data = np.fromfile(file, dtype=np.uint32) 
        # Extract lower 16 bits for the label
        label_values = data & 0xFFFF
    
        # Extract upper 16 bits for the instance id
        instance_ids = data >> 16
    
        return label_values, instance_ids
        




file_path = '/mnt/ssd2/jihun/dataset/sequences/00/labels/001000.label'

labels, instance_ids = load_label(file_path)
i = 100000
print(labels.shape)
unique_labels = np.unique(labels)
print(unique_labels)
print(unique_labels.shape)

file_path = '/mnt/ssd2/jihun/dataset/sequences/00/velodyne/000000.bin'

pc_data = load_pc(file_path)
print(pc_data.shape)
