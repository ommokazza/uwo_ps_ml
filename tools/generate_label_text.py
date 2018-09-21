"""This generate label file from screenshot image by using learned model.
"""
import os

import tensorflow as tf
from tensorflow.saved_model import tag_constants

from uwo_ps_utils import common
from uwo_ps_utils import market_rates_cropper as mrc

def main(screenshot_dir, models, labels, filter_str, label_fix_func):
    """
    Arguments:
        screenshot_dir: The screenshot directory
        models : The model directory list. [goods, towns, rates, arrows]
        labels : The label file list. [goods, towns, rates, arrows]
    """

    goods_labels = open(labels[0]).read().splitlines()
    towns_labels = open(labels[1]).read().splitlines()
    rates_labels = open(labels[2]).read().splitlines()
    arrows_labels = open(labels[3]).read().splitlines()

    image_paths = common.get_image_paths(screenshot_dir, "bmp")
    for path in image_paths:
        if __filtered(path, filter_str):
            continue

        print("Processing...", path)
        labels = []
        if not path.lower().endswith(".bmp"):
            continue
        images = mrc.get_images_from_screenshot(path)

        for i in range(0, len(images), 3):
            if i == 0:
                resize_ratio = 6
                im = images[0].resize((int(images[0].width / resize_ratio),
                                       int(images[0].height / resize_ratio)))
                index = common.estimate(models[0], im.tobytes())
                labels.append(goods_labels[index])
            else:
                index = common.estimate(models[1], images[i].tobytes())
                labels.append(towns_labels[index])

            index = common.estimate(models[2], images[i+1].tobytes())
            labels.append(rates_labels[index])
            index = common.estimate(models[3], images[i+2].tobytes())
            labels.append(arrows_labels[index])

        labels = label_fix_func(labels)
        with open(path[:-4] + ".txt", "w") as f:
            f.write('\n'.join(labels) + '\n')

def __filtered(path, filter_str):
    if filter_str == "":
        return False

    return not os.path.dirname(path).endswith(filter_str)