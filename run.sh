#!/bin/bash

# Set the environment variable to disable Mesa drivers
# export OPENCV_VIDEOIO_PRIORITY_MSMF=0

# Activate the 'ges' conda environment
source ~/miniconda3/bin/activate ges

# Run the main.py script
python main.py

# Deactivate the conda environment
conda deactivate
