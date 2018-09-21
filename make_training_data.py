#! python3
"""A Script to run generate_training_data.py
"""
import os
# import tools.generate_training_data as gtd

# Modify This --------------------------------------------------
# "traning_data_goods" is not used by default.
# we have alread enough data from ivyro site
# You have to copy generated files to resources directory
# to apply new training data to machine learning.
SCREENSHOT_DIR = "./screenshots"
OUTPUT_DIRS = [
    "",#"./gen/training_data_goods",
    "./gen/training_data_towns",
    "./gen/training_data_rates",
    "",#./gen/training_data_arrows"
    ]
# --------------------------------------------------------------

cmd = "python tools/generate_training_data.py"
cmd += " --screenshot_dir=" + os.path.abspath(SCREENSHOT_DIR)
if OUTPUT_DIRS[0] != "":
    cmd += " --goods_dir=" + os.path.abspath(OUTPUT_DIRS[0])
if OUTPUT_DIRS[1] != "":
    cmd += " --towns_dir=" + os.path.abspath(OUTPUT_DIRS[1])
if OUTPUT_DIRS[2] != "":
    cmd += " --rates_dir=" + os.path.abspath(OUTPUT_DIRS[2])
if OUTPUT_DIRS[3] != "":
    cmd += " --arrows_dir=" + os.path.abspath(OUTPUT_DIRS[3])

os.system(cmd)