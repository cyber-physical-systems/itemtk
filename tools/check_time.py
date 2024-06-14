import os
import pandas as pd
import random

dataset_directory = 'yourdataset/data_csv'


start_times = []


for root, dirs, files in os.walk(dataset_directory):
    for file in files:
        if file.endswith('.csv'):

            file_path = os.path.join(root, file)

            try:
                df = pd.read_csv(file_path, nrows=1)

                start_time = df.iloc[0][0]

                category = os.path.basename(root)
                start_times.append((start_time, category, file))
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

start_times.sort()



import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
from datetime import datetime, timedelta
import shutil

start_times = [(datetime.strptime(time, '%Y-%m-%d %H:%M:%S'), category, file) for time, category, file in start_times]
start_times.sort(key=lambda x: x[0])

original_folder = 'yourdataset/data_original_csv'
destination_folder = 'yourdataset/data_destination_csv'

specific_start_time = datetime(2016, 10, 3, 15, 0, 0)
end_time_window = specific_start_time + timedelta(hours=2)
selected_times = [item for item in start_times if specific_start_time <= item[0] <= end_time_window]

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)


for time, category, file in start_times:
    if specific_start_time <= time <= end_time_window:

        category_folder = os.path.join(destination_folder, category)
        if not os.path.exists(category_folder):
            os.makedirs(category_folder)

        original_file_path = os.path.join(original_folder, category, file)
        destination_file_path = os.path.join(category_folder, file)


        try:
            shutil.copy(original_file_path, destination_file_path)
            print(f"file {file} has been copied to {destination_file_path}")
        except IOError as e:
            print(f"can not copy {file}. for: {e}")

plt.figure(figsize=(50, 6))
ax = plt.subplot(1, 1, 1)

for i, (time, category, file) in enumerate(selected_times):
    plt.plot([time, time], [0, 1], color='blue')
    plt.text(time, 1, f'{i}\n({category})', fontsize=8, rotation=45, ha='right')

ax.xaxis.set_major_locator(mdates.HourLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))


plt.title('File Start Times within 10 Hours')
plt.xlabel('Time')
plt.ylabel('File Number')

plt.tight_layout()
plt.savefig('/home/activity_data.png')  #

