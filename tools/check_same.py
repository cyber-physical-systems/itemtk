import os


def compare_image_files(directory1, directory2):
    mismatched_files = []

    # Check if both directories exist
    if not os.path.isdir(directory1) or not os.path.isdir(directory2):
        print("One or both directories do not exist.")
        return None

    subdirs1 = set(os.listdir(directory1))
    subdirs2 = set(os.listdir(directory2))

    if subdirs1 != subdirs2:
        print("Warning: Subdirectories do not match between the two directories.")
        return None

    # Compare the image files in each subdirectory
    for subdir in subdirs1:
        subdir_path1 = os.path.join(directory1, subdir)
        subdir_path2 = os.path.join(directory2, subdir)

        if not os.path.isdir(subdir_path1) or not os.path.isdir(subdir_path2):
            continue

        image_files1 = {file for file in os.listdir(subdir_path1) if file.endswith('.png')}
        image_files2 = {file for file in os.listdir(subdir_path2) if file.endswith('.png')}

        if image_files1 != image_files2:
            # Find differences
            differences = image_files1.symmetric_difference(image_files2)
            for file in differences:
                if file in image_files1:
                    mismatched_files.append(os.path.join(subdir_path1, file))
                else:
                    mismatched_files.append(os.path.join(subdir_path2, file))

    return mismatched_files


# Example usage:
directory1 = '/yourdataset/ScatterPlots_reshape'
directory2 = '/yourdataset/GAF_reshape'
mismatches = compare_image_files(directory1, directory2)
if mismatches:
    for file_path in mismatches:
        print(f'Mismatched file: {file_path}')
else:
    print("All files match between the directories.")



