#!/usr/bin/env python
"""
List all files in the root directory and delete a file by index.
"""

import os
from pathlib import Path

# Define the root directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# List all files in the root directory
files = [f for f in os.listdir(ROOT_DIR) if os.path.isfile(os.path.join(ROOT_DIR, f))]

# Print the files with indices
print("Files in the root directory:")
for i, file in enumerate(files):
    print(f"{i}: {file}")

# Ask for the index of the file to delete
index = int(input("Enter the index of the file to delete (or -1 to cancel): "))

if index >= 0 and index < len(files):
    file_to_delete = files[index]
    file_path = os.path.join(ROOT_DIR, file_to_delete)
    
    # Confirm deletion
    confirm = input(f"Are you sure you want to delete '{file_to_delete}'? (y/n): ")
    
    if confirm.lower() == 'y':
        try:
            os.remove(file_path)
            print(f"File '{file_to_delete}' deleted successfully")
        except Exception as e:
            print(f"Error deleting file: {str(e)}")
    else:
        print("Deletion cancelled")
elif index == -1:
    print("Deletion cancelled")
else:
    print("Invalid index")

print("Done")
