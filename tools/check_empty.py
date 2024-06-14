import os
import pandas as pd

def find_and_delete_empty_totalin_csvs(directory):

    empty_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:

            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                try:

                    df = pd.read_csv(file_path)

                    if 'TotalIn' not in df.columns or df['TotalIn'].isna().all():
                        empty_files.append(file_path)
                        os.remove(file_path)
                        print(f"Deleted empty TotalIn CSV: {file_path}")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    return empty_files


your_directory_path = '/home/your/dataset_csv'

empty_csvs = find_and_delete_empty_totalin_csvs(your_directory_path)
print(f"Total empty TotalIn CSVs deleted: {len(empty_csvs)}")
