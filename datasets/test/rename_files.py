import os

def rename_files(folder_path):
    for filename in os.listdir(folder_path):
        original_path = os.path.join(folder_path, filename)
        if os.path.isfile(original_path):
            new_filename = filename.split("-", 1)[-1]  # Extract string after "-"
            new_path = os.path.join(folder_path, new_filename)
            os.rename(original_path, new_path)
            print(f"Renamed {filename} to {new_filename}")


# Call the function to rename files in the folder
rename_files("norain")
rename_files("rain")
