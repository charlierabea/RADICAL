#!/bin/bash

# Define the source (which was the previous destination) and destination (which was the previous source) directories
SOURCE_DIR="/home/chengyili/data/CT/test2/"
DEST_DIR="/home/chengyili/data/CT/test/"

# Move all .bmp files from the source directory to the destination directory
find "$SOURCE_DIR" -type f -name "*.bmp" -exec mv {} "$DEST_DIR" \;
