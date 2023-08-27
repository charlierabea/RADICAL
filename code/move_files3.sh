#!/bin/bash

# Define the source and destination directories
SOURCE_DIR="/home/chengyili/data/CT/mved/"
DEST_DIR="/home/chengyili/data/CT/evaluate/"
LOG_FILE="script_log.txt"

# Log the start of the process
echo "Starting the script at $(date)" >> "$LOG_FILE"

# Check if destination directory exists, if not, create it
if [ ! -d "$DEST_DIR" ]; then
    echo "Destination directory does not exist. Creating $DEST_DIR..." >> "$LOG_FILE"
    mkdir -p "$DEST_DIR"
fi

# Get a list of all unique studies
studies=$(find "$SOURCE_DIR" -type f -name "*.bmp" | sed -e 's/.*\/\(.*\)_\([0-9]*\)\.bmp$/\1/' | sort | uniq)

# Shuffle the studies and take 20% of them
selected_studies=$(echo "$studies" | shuf | head -n $(echo "$studies" | wc -l | awk '{print int($1*0.2)}'))

echo "Selected studies for moving:" >> "$LOG_FILE"
echo "$selected_studies" >> "$LOG_FILE"

# Move the .bmp files of the selected studies to the test directory
IFS=$'\n'  # Change the internal field separator to newline for the for loop
for study in $selected_studies; do
    echo "Moving files for study: $study" >> "$LOG_FILE"
    find "$SOURCE_DIR" -type f -name "${study}_*.bmp" -exec mv {} "$DEST_DIR" \;
done

# Log the completion of the process
echo "Script completed at $(date)" >> "$LOG_FILE"
