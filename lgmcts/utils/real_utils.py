"""Utils for realworld robot experiment"""
from __future__ import annotations

import ast
import numpy as np

# Initialize variables to store data
grasping_height = []
label = []
polygons = []
boundary = None
camera_pose = None
intrinsics = None


def load_env_data(file_name: str):
    """Read the file line by line"""
    with open(file_name, 'r') as file:
        lines = file.readlines()

    # Define a flag to determine the current section
    current_section = None
    grasping_height = []
    label = []
    polygons = []
    boundary = None
    camera_pose = None
    intrinsics = None

    # Loop through the lines and process the data
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace

        # Check if the line is empty or starts with a comment
        if not line or line.startswith('#'):
            continue

        # Check if the line indicates a new section
        if line.startswith('# grasping height'):
            current_section = 'grasping_height'
        elif line.startswith('# label'):
            current_section = 'label'
        elif line.startswith('# polygons'):
            current_section = 'polygons'
            polygons_data = []
        elif line.startswith('# workspace'):
            current_section = 'workspace'
        elif line.startswith('# camera'):
            current_section = 'camera_pose'
        elif line.startswith('intrinsics'):
            current_section = 'intrinsics'

        # Process lines based on the current section
        elif current_section == 'grasping_height':
            grasping_height = ast.literal_eval(line)
        elif current_section == 'label':
            label = ast.literal_eval(line)
        elif current_section == 'polygons':
            # Read polygons data until a blank line is encountered
            if line:
                polygon = ast.literal_eval(line)
                polygons_data.append(polygon)
            else:
                polygons.append(polygons_data)
        elif current_section == 'workspace':
            boundary = ast.literal_eval(line)
        elif current_section == 'camera_pose':
            camera_pose = np.array(ast.literal_eval(line))
        elif current_section == 'intrinsics':
            intrinsics = np.array(ast.literal_eval(line))

    return grasping_height, label, polygons, boundary, camera_pose, intrinsics
