import os
import random

def delete_specific_csv_files(directory, delete_rates, seed=0):
    random.seed(seed)

    for subdir, delete_rate in delete_rates.items():
        subdir_path = os.path.join(directory, subdir)

        if os.path.exists(subdir_path) and os.path.isdir(subdir_path):

            csv_files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith('.png')]

            num_files_to_delete = int(len(csv_files) * delete_rate)
            files_to_delete = random.sample(csv_files, num_files_to_delete)
            for file_path in files_to_delete:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")

dataset_path = "/your_dataset/device_injection_SCA"

delete_rates = {
    'Amazon': 0.6,
    'Insteon': 0.4,
    'Welcome': 0.4
}

delete_specific_csv_files(dataset_path, delete_rates, seed=0)
