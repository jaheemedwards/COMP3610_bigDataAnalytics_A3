import os
import subprocess
from pathlib import Path

def set_full_control(folder_path):
    """Grant full control permissions to a folder."""
    try:
        # Use icacls to grant Full Control to everyone
        subprocess.run(
            ['icacls', folder_path, '/grant', 'Everyone:F', '/T'], 
            check=True
        )
        print(f"Full control granted for {folder_path}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to set permissions for {folder_path}. Error: {e}")

def create_project_directories(base_path):
    """Create necessary project directories under the 'data' directory if they do not exist."""
    # Paths to the three directories under 'data'
    data_directories = [
        'data/raw', 
        'data/processed/models', 
        'data/logs'
    ]
    
    for dir_name in data_directories:
        dir_path = os.path.join(base_path, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
        else:
            print(f"Directory already exists: {dir_path}")
    
    # Return the path to 'data' so permissions can be set on all subdirectories
    return os.path.join(base_path, 'data')

def initialize_project():
    """Initialize the project by creating directories and setting permissions."""
    base_path = os.getcwd()  # Get the current working directory
    print(f"Initializing project in: {base_path}")

    # Step 1: Create the directories (raw, processed, logs inside 'data')
    data_folder = create_project_directories(base_path)
    
    # Step 2: Set full control permissions for the 'data' folder and its subdirectories
    set_full_control(data_folder)

if __name__ == "__main__":
    initialize_project()
