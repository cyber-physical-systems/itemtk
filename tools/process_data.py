import os
import pandas as pd

def process_directory(directory_path):
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(directory_path, file_name)
            process_csv(file_path)

def process_csv(file_path):
    df = pd.read_csv(file_path)
    if set(['TIME', 'TotalIn', 'TotalOut']).issubset(df.columns):

        df = df[['TIME', 'TotalIn', 'TotalOut']]

        df.to_csv(file_path, index=False)
        print(f"Processed file: {file_path}")
    else:
        print(f"Skipped file (missing columns): {file_path}")

if __name__ == "__main__":

    your_data_folder = "/home/your_data/data/groundtruth_original_2000_csv"

    for subdir in os.listdir(your_data_folder):
        subdir_path = os.path.join(your_data_folder, subdir)

        if os.path.isdir(subdir_path):
            process_directory(subdir_path)
