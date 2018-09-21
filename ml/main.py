import argparse

from learning_model_goods import GoodsLearningModel
from learning_model_towns import TownsLearningModel
from learning_model_rates import RatesLearningModel
from learning_model_arrows import ArrowsLearningModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--choices")
    parser.add_argument("-i", "--input_dir")
    parser.add_argument("-m", "--model_dir")
    parser.add_argument("-l", "--label_path")
    args = parser.parse_args()

    output = "Start ML - %s"
    if args.choices == "1":
        print(output % "Trade Goods")
        model = GoodsLearningModel(args.input_dir,
                                   args.model_dir,
                                   args.label_path)
    elif args.choices == "2":
        print(output % "Nearby Towns")
        model = TownsLearningModel(args.input_dir,
                                   args.model_dir,
                                   args.label_path)
    elif args.choices == "3":
        print(output % "Rates")
        model = RatesLearningModel(args.input_dir,
                                   args.model_dir,
                                   args.label_path)
    elif args.choices == "4":
        print(output % "Arrows")
        model = ArrowsLearningModel(args.input_dir,
                                    args.model_dir,
                                    args.label_path)
    else:
        print("Wrong choice: ", args.choices)

    model.learn()