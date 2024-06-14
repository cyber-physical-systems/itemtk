import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from pyts.image import GramianAngularField
import pandas as pd
import os
from typing import *

matplotlib.use('Agg')


# Pass times-eries and create a Gramian Angular Field image
# Grab times-eries and draw the charts
def create_gaf(ts) -> Dict[str, Any]:
    """
    :param ts:
    :return:
    """
    data = dict()
    gadf = GramianAngularField(method='difference', image_size=ts.shape[0])
    data['gadf'] = gadf.fit_transform(pd.DataFrame(ts).T)[0]
    return data




def create_images(X_plots, image_name, folder, image_matrix=(2, 2)):

    destination_path = os.path.join(OUTPUT_PATH, folder)
    print(f"Saving images to {destination_path}")

    os.makedirs(destination_path, exist_ok=True)

    image_path = os.path.join(destination_path, image_name)

    fig, axes = plt.subplots(nrows=image_matrix[0], ncols=image_matrix[1], figsize=(8, 8))
    for ax, data in zip(axes.flatten(), X_plots):
        ax.imshow(data, cmap='rainbow', origin='lower')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()
    print(f"Image saved as {image_path}")

