import os
import numpy as np
from cellpose import plot
import matplotlib.pyplot as plt

def create_directories(base_path):
    result_dir = os.path.join(base_path, 'results')
    each_image_dir = os.path.join(result_dir, 'EachImage')
    os.makedirs(each_image_dir, exist_ok=True)
    return result_dir, each_image_dir

def show_segmentation(wga_channel, masks, flows):
    fig = plot.show_segmentation(
        plt.figure(figsize=(20, 10)),
        wga_channel,
        masks,
        flows[0],
        channels=[0, 0]
    )
    plt.show()