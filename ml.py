#! python3
"""Start machine learning
"""
import os

INPUT_DIRS = [""] * 5
MODEL_DIRS = [""] * 5
LABEL_PATHS = [""] * 5

INPUT_DIRS[1] = "./resources/training_data_goods"
MODEL_DIRS[1] = "./gen/model_goods"
LABEL_PATHS[1] = "./gen/goods.labels"

INPUT_DIRS[2] = "./resources/training_data_towns"
MODEL_DIRS[2] = "./gen/model_towns"
LABEL_PATHS[2] = "./gen/towns.labels"

INPUT_DIRS[3] = "./resources/training_data_rates"
MODEL_DIRS[3] = "./gen/model_rates"
LABEL_PATHS[3] = "./gen/rates.labels"

INPUT_DIRS[4] = "./resources/training_data_arrows"
MODEL_DIRS[4] = "./gen/model_arrows"
LABEL_PATHS[4] = "./gen/arrows.labels"


if __name__ == "__main__":
    print("Select learning models to run(With space separator):")
    print("  1. Trade Goods")
    print("  2. Nearby Towns")
    print("  3. Rates")
    print("  4. Arrows")
    choices = input("Choice? ").split(" ")

    for c in choices:
        index = int(c)
        if not (1 <= index and index <= 4):
            print("Wrong choice:", c)
            continue

        cmd = "python ml/main.py"
        cmd += " --choices=" + c
        cmd += " --input_dir=" + INPUT_DIRS[index]
        cmd += " --model_dir=" + MODEL_DIRS[index]
        cmd += " --label_path=" + LABEL_PATHS[index]

        os.system(cmd)