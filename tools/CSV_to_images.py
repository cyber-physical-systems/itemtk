import os
import time
import pandas as pd
import psutil
from scipy.fft import fft
from scipy.signal import stft
import pywt

DATA_PATH = "./data/groundtruth_original_2000_csv"
OUTPUT_PATH = "./activity_data/data"

import matplotlib.pyplot as plt
import os


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product





def set_single_heatmap_single_day(df, csv_name, folder):
    print(f'Generating single heatmap for {csv_name} in folder {folder}')

    OUTPUT_PATH_1 = os.path.join(OUTPUT_PATH, 'original_INO_in')
    output_dir = os.path.join(OUTPUT_PATH_1, folder)
    os.makedirs(output_dir, exist_ok=True)

    granularity = 1
    df_resampled = df.resample(f'{granularity}S').mean().dropna()

    heatmap_data = df_resampled['TotalIn'].values.reshape(1, -1)

    plt.imshow(heatmap_data, aspect='auto', cmap='hot')

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    plt.tight_layout(pad=0)

    heatmap_image_name = f"{os.path.splitext(csv_name)[0]}.png"
    heatmap_image_path = os.path.join(output_dir, heatmap_image_name)


    plt.savefig(heatmap_image_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    print(f"Heatmap saved as {heatmap_image_path}")

def set_wavelet_transform_chart(df, csv_name, folder):

    print(f'Generating wavelet transform chart for {csv_name} in folder {folder}')

    OUTPUT_PATH_1 = os.path.join(OUTPUT_PATH, 'injection_WAV_in')
    output_dir = os.path.join(OUTPUT_PATH_1, folder)
    os.makedirs(output_dir, exist_ok=True)

    wavelet_chart_name = f"{os.path.splitext(csv_name)[0]}_wavelet.png"


    wavelet = 'cmor'
    scales = np.arange(1, 128)


    signal = df['TotalIn'].dropna().values
    coefficients, frequencies = pywt.cwt(signal, scales, wavelet)


    plt.figure(figsize=(10, 6))
    plt.imshow(abs(coefficients), extent=[0, len(signal), 1, max(scales)], cmap='jet', aspect='auto',
               vmax=abs(coefficients).max(), vmin=-abs(coefficients).max())


    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)


    plt.savefig(os.path.join(output_dir, wavelet_chart_name), bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Wavelet Transform chart saved as {os.path.join(output_dir, wavelet_chart_name)}")


def set_INOUT_line_data_single_day(df, csv_name, folder):
    print(f'Generating line images for {csv_name} in folder {folder}')

    granularity = 1
    OUTPUT_PATH_1 = os.path.join(OUTPUT_PATH, 'padding_LIN_out')
    output_dir = os.path.join(OUTPUT_PATH_1, folder)


    os.makedirs(output_dir, exist_ok=True)
    line_image_name = f"{os.path.splitext(csv_name)[0]}.png"

    df_resampled = df.resample(f'{granularity}S').mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(df_resampled.index, df_resampled['TotalIn'], label='Total In', marker='o', color='C0')
    ax.plot(df_resampled.index, df_resampled['TotalOut'], label='Total Out', marker='o', color='C1')
    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    ax.legend().set_visible(False)

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    line_image_path = os.path.join(output_dir, line_image_name)

    fig.savefig(line_image_path, bbox_inches='tight', pad_inches=0, transparent=False)
    plt.close(fig)

    print(f"Line plot saved as {line_image_path}")

def set_line_data_single_day(df, csv_name, folder):
    print(f'Generating line images for {csv_name} in folder {folder}')

    granularity = 1
    OUTPUT_PATH_1 = os.path.join(OUTPUT_PATH, 'paros_LIN')
    output_dir = os.path.join(OUTPUT_PATH_1, folder)
    # df.set_index('TIME', inplace=True)

    os.makedirs(output_dir, exist_ok=True)
    line_image_name = f"{os.path.splitext(csv_name)[0]}.png"
    df_resampled = df.resample(f'{granularity}S').mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_resampled.index, df_resampled['TotalIn'], label='Total In', marker='o', color='C0')
    ax.plot(df_resampled.index, df_resampled['TotalOut'], label='Total Out', marker='o', color='C1')
    ax.set_facecolor('white')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    ax.legend().set_visible(False)

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    line_image_path = os.path.join(output_dir, line_image_name)

    fig.savefig(line_image_path, bbox_inches='tight', pad_inches=0, transparent=False)
    plt.close(fig)

    print(f"Line plot saved as {line_image_path}")


def set_inoutput_heatmap_single_day(df, csv_name, folder):
    print(f'Generating inoutput images for {csv_name} in folder {folder}')

    granularity = 1
    OUTPUT_PATH_1 = os.path.join(OUTPUT_PATH, 'paros_INO')
    output_dir = os.path.join(OUTPUT_PATH_1, folder)
    os.makedirs(output_dir, exist_ok=True)

    df_resampled = df.resample(f'{granularity}S').mean().dropna()

    heatmap_data = np.zeros((2, len(df_resampled)))
    heatmap_data[0, :] = df_resampled['TotalIn']
    heatmap_data[1, :] = df_resampled['TotalOut']

    plt.imshow(heatmap_data, aspect='auto', cmap='hot')

    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    plt.tight_layout(pad=0)

    heatmap_image_name = f"{os.path.splitext(csv_name)[0]}.png"

    heatmap_image_path = os.path.join(output_dir, heatmap_image_name)

    plt.savefig(heatmap_image_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    print(f"Heatmap saved as {heatmap_image_path}")


def set_spectrum_heatmap_single_day(df, csv_name, folder):
    print(f'Generating spectrum heatmaps for {csv_name} in folder {folder}')

    OUTPUT_PATH_1 = os.path.join(OUTPUT_PATH, 'pg_SPE_HEA')
    output_dir = os.path.join(OUTPUT_PATH_1, folder)
    os.makedirs(output_dir, exist_ok=True)

    spectrum_heatmap_name = f"{os.path.splitext(csv_name)[0]}.png"

    signal_in = df['TotalIn'].values
    signal_out = df['TotalOut'].values

    f_in, t_in, Zxx_in = stft(signal_in, nperseg=256)
    f_out, t_out, Zxx_out = stft(signal_out, nperseg=256)

    fig, axs = plt.subplots(2, 1, figsize=(6, 6))

    axs[0].pcolormesh(t_in, f_in, np.abs(Zxx_in), shading='gouraud')
    axs[0].axis('off')

    axs[1].pcolormesh(t_out, f_out, np.abs(Zxx_out), shading='gouraud')
    axs[1].axis('off')

    spectrum_heatmap_path = os.path.join(output_dir, spectrum_heatmap_name)

    plt.tight_layout(pad=0)
    plt.savefig(spectrum_heatmap_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Spectrum heatmap saved as {spectrum_heatmap_path}")


def set_scatter_data_single_day(df, csv_name, folder):

    print(f'Generating scatter images for {csv_name} in folder {folder}')

    OUTPUT_PATH_1 = os.path.join(OUTPUT_PATH, 'paros_SCA')
    output_dir = os.path.join(OUTPUT_PATH_1, folder)
    os.makedirs(output_dir, exist_ok=True)

    scatter_image_name = f"{os.path.splitext(csv_name)[0]}.png"

    granularity = 1

    plt.figure(figsize=(6, 6))
    plt.scatter(df_resampled['TotalIn'], df_resampled['TotalOut'], alpha=0.7, color='black',edgecolors='w', s=80)

    plt.axis('off')

    scatter_image_path = os.path.join(output_dir, scatter_image_name)

    plt.savefig(scatter_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"Scatter plot saved as {scatter_image_path}")

def set_16gaf_data_single_day(df, csv_name, folder):
    print(f'Generating 16 gaf images for {csv_name} in folder {folder}')

    OUTPUT_PATH1 = os.path.join(OUTPUT_PATH, 'GAF_16')
    output_dir = os.path.join(OUTPUT_PATH1, folder)
    os.makedirs(output_dir, exist_ok=True)

    freqs = [f'{i}S' for i in range(1, 17)]
    gafs = []
    for freq in freqs:
        resampled_df = df.resample(freq).mean().dropna()
        gaf_image = generate_gaf_image(resampled_df['TotalIn'].values)
        gafs.append(gaf_image)

    image_name = f"{os.path.splitext(csv_name)[0]}.png"
    save_gaf_16_images(gafs, image_name, output_dir)

    print(f"Processed {csv_name}: Total Data Points: {len(df)}")

def set_device_16gaf_data_single_day(df: pd.DataFrame, csv_name: str, folder: str) -> None:
    print(f'Generating 16 gaf images for {csv_name} in folder {folder}')

    OUTPUT_PATH1 = './your/data/foler/GAF_16'
    output_dir = os.path.join(OUTPUT_PATH1, folder)
    os.makedirs(output_dir, exist_ok=True)

    freqs = [
        '10S', '15S', '30S', '45S', '1T',  '2T',  '3T',
        '4T', '5T', '10T', '15T', '20T', '30T', '45T', '60T','75T'
    ]
    intervals = pd.date_range(start=df.index.min(), end=df.index.max(), freq='8H')

    for idx, start_time in enumerate(intervals[:-1]):
        end_time = intervals[idx + 1]
        slice_df = df[start_time:end_time]

        gafs = []
        for freq in freqs:
            resampled_df = slice_df.resample(freq).mean().dropna()
            column_name = folder + 'In'
            if column_name in resampled_df.columns:
                gaf_image = generate_gaf_image(resampled_df[column_name].values)
                gafs.append(gaf_image)
            else:
                print(f"Column {column_name} not found in DataFrame.")

        if len(gafs) == 16:
            image_name = f"{os.path.splitext(csv_name)[0]}_{idx}.png"
            save_gaf_16_images(gafs, image_name, output_dir)
        else:
            print(f"Error: Not enough GAF images for interval {idx}. Expected 16, got {len(gafs)}.")

    print(f"Processed {csv_name}: Total Data Points: {len(df)}")

def set_9gaf_data_single_day(df, csv_name, folder):
    print(f'Generating 9 gaf images for {csv_name} in folder {folder}')

    OUTPUT_PATH1 = os.path.join(OUTPUT_PATH, 'padding_GAF_9')
    output_dir = os.path.join(OUTPUT_PATH1, folder)
    os.makedirs(output_dir, exist_ok=True)

    freqs = ['1S', '2S', '3S', '4S', '5S', '6S', '7S', '8S', '9S']
    gafs = []
    for freq in freqs:
        resampled_df = df.resample(freq).mean().dropna()
        gaf_image = generate_gaf_image(resampled_df['TotalIn'].values)
        gafs.append(gaf_image)

    image_name = f"{os.path.splitext(csv_name)[0]}.png"
    save_gaf_9_images(gafs, image_name, output_dir,dpi=150)

    print(f"Processed {csv_name}: Total Data Points: {len(df)}")

def set_device_9gaf_data_single_day(df: pd.DataFrame, csv_name: str, folder: str) -> None:
    print(f'Generating 9 gaf images for {csv_name} in folder {folder}')

    OUTPUT_PATH1 = '/your/data/folder/GAF_9'
    output_dir = os.path.join(OUTPUT_PATH1, folder)
    os.makedirs(output_dir, exist_ok=True)

    freqs = ['30S', '1T', '2T', '5T', '10T', '15T', '20T', '30T', '60T']
    intervals = pd.date_range(start=df.index.min(), end=df.index.max(), freq='8H')

    for idx, start_time in enumerate(intervals[:-1]):
        end_time = intervals[idx + 1]
        slice_df = df[start_time:end_time]

        gafs = []
        for freq in freqs:
            resampled_df = slice_df.resample(freq).mean().dropna()
            column_name = folder + 'In'
            gaf_image = generate_gaf_image(resampled_df[column_name].values)
            gafs.append(gaf_image)

        if len(gafs) == 9:
            image_name = f"{os.path.splitext(csv_name)[0]}_{idx}.png"
            save_gaf_9_images(gafs, image_name, output_dir)
        else:
            print(f"Error: Not enough GAF images for interval {idx}. Expected 9, got {len(gafs)}.")

    print(f"Processed {csv_name}: Total Data Points: {len(df)}")


def set_device_4gaf_data_single_day(df: pd.DataFrame, csv_name: str, folder: str) -> None:
    print(f'Generating 4 gaf images for {csv_name} in folder {folder}')

    OUTPUT_PATH1 = '/your/data/folder/GAF_reshape'
    output_dir = os.path.join(OUTPUT_PATH1, folder)
    os.makedirs(output_dir, exist_ok=True)

    freqs = ['1T', '5T', '15T', '30T']
    intervals = pd.date_range(start=df.index.min(), end=df.index.max(), freq='8H')

    for idx, start_time in enumerate(intervals[:-1]):
        end_time = intervals[idx + 1]
        slice_df = df[start_time:end_time]

        gafs = []
        for freq in freqs:
            resampled_df = slice_df.resample(freq).mean().dropna()
            column_name = folder + 'In'
            if column_name in resampled_df.columns:
                gaf_image = generate_gaf_image(resampled_df[column_name].values)
                gafs.append(gaf_image)
            else:
                print(f"Column {column_name} not found in DataFrame.")

        if len(gafs) == 4:
            image_name = f"{os.path.splitext(csv_name)[0]}_{idx}.png"
            save_gaf_images(gafs, image_name, output_dir)
        else:
            print(f"Error: Not enough GAF images for interval {idx}. Expected 4, got {len(gafs)}.")

    print(f"Processed {csv_name}: Total Data Points: {len(df)}")

def set_gaf_data_single_day(df, csv_name, folder):
    print(f'Generating gaf images for {csv_name} in folder {folder}')
    # df.set_index('TIME', inplace=True)
    memory_usage = psutil.virtual_memory()
    print(f"Memory Usage: {memory_usage.percent}%")

    freqs = ['1S', '2S', '5S', '10S']
    image_suffixes = ['01', '02', '03', '04']
    OUTPUT_PATH1 = os.path.join(OUTPUT_PATH, 'GAF')
    output_dir = os.path.join(OUTPUT_PATH1, folder)

    os.makedirs(output_dir, exist_ok=True)

    gafs = []
    for i, freq in enumerate(freqs):
        resampled_df = df.resample(freq).mean().dropna()

        gaf_image = generate_gaf_image(resampled_df['TotalIn'].values)

        gafs.append(gaf_image)

    image_name = f"{os.path.splitext(csv_name)[0]}.png"
    save_gaf_images(gafs, image_name, output_dir)

    print(f"Processed {csv_name}: Total Data Points: {len(df)}")

def set_single_gaf_image(df, csv_name, folder):
    print(f'Generating single gaf image for {csv_name} in folder {folder}')

    OUTPUT_PATH1 = os.path.join(OUTPUT_PATH, 'paros_GAF_1')
    output_dir = os.path.join(OUTPUT_PATH1, folder)

    os.makedirs(output_dir, exist_ok=True)

    freq = '1S'
    resampled_df = df.resample(freq).mean().dropna()
    gaf_image = generate_gaf_image(resampled_df['TotalOut'].values)

    image_name = f"{os.path.splitext(csv_name)[0]}.png"

    save_gaf_image(gaf_image, image_name, output_dir)


def set_device_1gaf_data_single_day(df: pd.DataFrame, csv_name: str, folder: str) -> None:
    print(f'Generating a single gaf image for {csv_name} in folder {folder}')

    OUTPUT_PATH1 = '/your/data/folder/GAF_1'
    output_dir = os.path.join(OUTPUT_PATH1, folder)
    os.makedirs(output_dir, exist_ok=True)

    freqs = ['15T']
    intervals = pd.date_range(start=df.index.min(), end=df.index.max(), freq='8H')

    for idx, start_time in enumerate(intervals[:-1]):
        end_time = intervals[idx + 1]
        slice_df = df[start_time:end_time]

        gafs = []
        for freq in freqs:
            resampled_df = slice_df.resample(freq).mean().dropna()
            column_name = folder + 'In'
            if column_name in resampled_df.columns:
                gaf_image = generate_gaf_image(resampled_df[column_name].values)
                gafs.append(gaf_image)
            else:
                print(f"Column {column_name} not found in DataFrame.")

        if len(gafs) == 1:
            image_name = f"{os.path.splitext(csv_name)[0]}_{idx}.png"
            save_gaf_1_image(gafs, image_name, output_dir)
        else:
            print(f"Error: Not enough GAF images for interval {idx}. Expected 1, got {len(gafs)}.")

    print(f"Processed {csv_name}: Total Data Points: {len(df)}")


from pyts.image import GramianAngularField
import numpy as np


def generate_gaf_image(time_series: np.ndarray) -> np.ndarray:
    size = 120

    if time_series.size < size:
        new_index = np.linspace(start=0, stop=time_series.size - 1, num=size)

        time_series = np.interp(new_index, np.arange(time_series.size), time_series)
    elif time_series.size > size:
        time_series = time_series[-size:]

    gadf = GramianAngularField(method='difference', image_size=size)
    gaf = gadf.fit_transform(time_series.reshape(1, -1))
    return gaf[0]

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def save_gaf_1_image(gaf_image, image_name, output_dir):
    gaf_image = np.squeeze(gaf_image)

    if gaf_image.ndim != 2:
        raise ValueError(
            f'gaf_image has an invalid number of dimensions after squeezing: {gaf_image.ndim}, expected 2.')

    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(gaf_image, cmap='rainbow', origin='lower')
    ax.axis('off')

    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, image_name), bbox_inches='tight', pad_inches=0)

    plt.close()

def save_gaf_image(gaf_image, image_name, output_dir):
    fig, ax = plt.subplots()
    ax.imshow(gaf_image, cmap='rainbow', origin='lower')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, image_name), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def save_gaf_images(gaf_images, image_name, output_dir):
    fig = plt.figure(figsize=(6, 6))
    grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0)

    for ax, gaf_image in zip(grid, gaf_images):
        ax.imshow(gaf_image, cmap='rainbow', origin='lower')
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, image_name), bbox_inches='tight', pad_inches=0)
    memory_usage = psutil.virtual_memory()
    print(f"Memory Usage: {memory_usage.percent}%")
    plt.close()

def save_gaf_16_images(gaf_images, image_name, output_dir):
    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axs.flat):
        ax.imshow(gaf_images[i], cmap='rainbow', origin='lower', aspect='auto')
        ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(output_dir, image_name), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def save_gaf_9_images(gaf_images, image_name, output_dir, dpi=150):
    fig = plt.figure(figsize=(9, 9))
    grid = ImageGrid(fig, 111, nrows_ncols=(3, 3), axes_pad=0)

    if len(gaf_images) != 9:
        print("Error: 9 GAF images are required.")
        return

    for ax, gaf_image in zip(grid, gaf_images):
        ax.imshow(gaf_image, cmap='rainbow', origin='lower')
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, image_name), bbox_inches='tight', pad_inches=0, dpi=dpi)  # 调整dpi
    plt.close()


if __name__ == "__main__":
    start_time = time.time()

    for folder in os.listdir(DATA_PATH):

        folder_path = os.path.join(DATA_PATH, folder)
        if os.path.isdir(folder_path):
            for csv_file in os.listdir(folder_path):
                if csv_file.endswith(".csv"):
                    file_path = os.path.join(folder_path, csv_file)

                    df = pd.read_csv(file_path)

                    df['TIME'] = pd.to_datetime(df['TIME'], infer_datetime_format=True)
                    if len(df) == 1440:
                        new_time = df.iloc[-1]['TIME'] + pd.Timedelta(minutes=1)
                        avg_in = df.iloc[:3][folder + 'In'].mean()
                        avg_out = df.iloc[:3][folder + 'Out'].mean()
                        df.loc[len(df)] = {'TIME': new_time, folder + 'In': avg_in, folder + 'Out': avg_out}

                    df.set_index('TIME', inplace=True)

                    set_single_heatmap_single_day(df, csv_file, folder)

    end_time = time.time()

    elapsed_time = end_time - start_time

    print(f"Program took {elapsed_time} seconds to run.")
