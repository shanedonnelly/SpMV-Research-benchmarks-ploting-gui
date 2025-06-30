import os
import tkinter as tk
from tkinter import filedialog
import subprocess

def main():
    """
    Opens a file dialog to select a CSV file and creates a symbolic link
    to it in the 'app/csv/' directory.
    """
    try:
        root = tk.Tk()
        root.withdraw()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        destination_folder = os.path.join(script_dir, 'app', 'csv')

        # Create the destination directory if it doesn't exist
        os.makedirs(destination_folder, exist_ok=True)

        source_file = filedialog.askopenfilename(
            title="Select a CSV file to link",
            filetypes=[("CSV Files", "*.csv")]
        )

        if not source_file:
            # User cancelled the file dialog
            return

        destination_path = os.path.join(destination_folder, os.path.basename(source_file))

        # Use subprocess to create the symbolic link via 'ln -s'
        result = subprocess.run(['ln', '-s', source_file, destination_path])

        if result.returncode == 0:
            print("Sucessfully created a symlink :")
            print(f"{os.path.abspath(source_file)} --> {os.path.abspath(destination_path)}")
        else:
            print("Error creating symlink")

    except Exception:
        print("Error creating symlink")

if __name__ == "__main__":
    main()