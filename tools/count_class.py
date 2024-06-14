import os


def count_csv_files_by_class(directory):
    class_csv_counts = {}

    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)

        if os.path.isdir(subdir_path):

            class_csv_counts[subdir] = 0

            for file in os.listdir(subdir_path):
                if file.endswith('.csv'):
                    class_csv_counts[subdir] += 1

    return class_csv_counts

directory_path = "/your_dataset/data_csv"

csv_counts = count_csv_files_by_class(directory_path)
for class_name, count in csv_counts.items():
    print(f"Class {class_name} has {count} CSV files")