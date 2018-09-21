#! python3
"""A Script to run generate_label_text.py
"""
import sys

import tools.generate_label_text as glt

# Modify This ------------------------------
SCREENSHOT_DIR = "./screenshots_no_label"
MODEL_DIRS = [
    "./resources/model_goods",
    "./resources/model_towns",
    "./resources/model_rates",
    "./resources/model_arrows"]
LABEL_PATHS = [
    "./resources/goods.labels",
    "./resources/towns.labels",
    "./resources/rates.labels",
    "./resources/arrows.labels"]
# ------------------------------------------

def __manual_fix(labels):
    # labels[3] = 'Ragusa'
    # labels[6] = 'Zadar'
    # labels[9] = 'Ancona'
    # labels[12] = 'Trieste'
    # labels[15] = 'Venice'
    return labels

glt.main(SCREENSHOT_DIR,
         MODEL_DIRS,
         LABEL_PATHS,
         "",    #filter string
         __manual_fix)
