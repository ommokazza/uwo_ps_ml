"""Make training data from screenshots and label files.

This will make 4 types of training data - goods, town, rate, arrow.
And each files are renamed arrording to labels

goods - {GoodsName}_{number}.png
towns - {TownName}_{number}.png
rate - {RateNumber}_{number}.png
arrow - {ArrowType}_{number}.png : 0 - Up, 1 - Neutral, 2 - Down

{number} is just for to prevent duplication name
"""
import argparse
import os
import sys

from functools import reduce
from PIL import Image

from uwo_ps_utils import market_rates_cropper as mrc
from uwo_ps_utils import common

def main(screenshot_dir, output_dirs):
    """Entry fucntion.

    Parameters:
        screenshot_dir: Full path of screenshot imagscreenshot_dirs and labels.
                        The text file of label has fscreenshot_dirllowing sequence lines:
                            name of trade goods
                            market rates of trade goscreenshot_dirds
                            arrow direction (Right-Uscreenshot_dir: 0, Right: 1, Right-Down: 2)
                            name of nearby town 1
                            market rates of nearby tscreenshot_dirwn 1
                            arrow direction of nearbscreenshot_dir town 1
                            [ iteration up to nearbyscreenshot_dirtown 5 ]

        output_dirs : The list of 4 output directoriscreenshot_dirs
                    [ goods_dirs, towns_dir, rates_dscreenshot_dirr, arrows_dir ]
    """
    files = common.get_image_paths(screenshot_dir, "bmp")
    for image_path in files:
        print('Processing...', image_path)
        label_path = os.path.splitext(image_path)[0] + ".txt"
        if not os.path.exists(label_path):
            print("No label file : ", label_path)
            continue

        __make_training_data(image_path, label_path, output_dirs)

def __make_training_data(image_path, label_path, output_dirs):
    if not __verify_path(image_path, label_path, output_dirs):
        return

    images = mrc.get_images_from_screenshot(image_path)
    with open(label_path) as f:
        lines = f.read().splitlines()

    name = image_path.split("/")[-1][:-4]
    name = os.path.splitext(os.path.basename(image_path))[0]
    output_format = "%s/%s_" + name + " %02d.png"
    output_sequences = [output_dirs[0], output_dirs[2], output_dirs[3],
                        output_dirs[1], output_dirs[2], output_dirs[3],
                        output_dirs[1], output_dirs[2], output_dirs[3],
                        output_dirs[1], output_dirs[2], output_dirs[3],
                        output_dirs[1], output_dirs[2], output_dirs[3],
                        output_dirs[1], output_dirs[2], output_dirs[3]]

    for i in range(len(images)):
        if output_sequences[i]:
            images[i].save(output_format % (output_sequences[i], lines[i], i))

    # For debugging to find wrong match
    if name == "":
        print(label_path, lines)
        #__show_images(images)

# Verify input files and make output directories
def __verify_path(image_path, label_path, output_dirs):
    if not os.path.exists(image_path):
        print("No image file : ", image_path)
        return False
    if not os.path.exists(label_path):
        print("No label file : ", label_path)
        return False

    for d in output_dirs:
        if d == "" or d == None or os.path.exists(d):
            continue
        os.makedirs(d)

    return True

# Debugging function
# This shows the cropped images in one.
def __show_images(images):
    test_img = Image.new("RGB", (100 * len(images), 17 * 25))
    for i in range(len(images)):
        for j in range(len(images[i])):
            test_img.paste(images[i][j], (i * 100, j * 25))

    test_img.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--screenshot_dir")
    parser.add_argument("-g", "--goods_dir")
    parser.add_argument("-t", "--towns_dir")
    parser.add_argument("-r", "--rates_dir")
    parser.add_argument("-a", "--arrows_dir")
    args = parser.parse_args()

    output_dirs = [
        args.goods_dir,
        args.towns_dir,
        args.rates_dir,
        args.arrows_dir
        ]

    main(args.screenshot_dir, output_dirs)