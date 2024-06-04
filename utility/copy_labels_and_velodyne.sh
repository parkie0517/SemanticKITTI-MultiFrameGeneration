#!/bin/bash

# Define base source and destination directories
BASE_SRC_DIR="/mnt/ssd2/jihun/dataset/sequences"
BASE_DEST_DIR="/mnt/ssd2/jihun/dataset_MF/sequences"

# Loop through the sequences from 00 to 10
for i in $(seq -w 0 10)
do
    # Define source and destination paths for labels
    SRC_LABELS="${BASE_SRC_DIR}/${i}/labels"
    DEST_LABELS="${BASE_DEST_DIR}/${i}/labels"
    
    # Define source and destination paths for velodyne
    SRC_VELODYNE="${BASE_SRC_DIR}/${i}/velodyne"
    DEST_VELODYNE="${BASE_DEST_DIR}/${i}/velodyne"
    
    # Create destination directories if they do not exist
    mkdir -p "$DEST_LABELS"
    mkdir -p "$DEST_VELODYNE"
    
    # Copy the labels and velodyne directories to the destination
    cp -r "$SRC_LABELS/"* "$DEST_LABELS/"
    cp -r "$SRC_VELODYNE/"* "$DEST_VELODYNE/"
    
    # Print the copy status
    echo "Copied sequence ${i} labels and velodyne data."
done

echo "All sequences copied successfully!"