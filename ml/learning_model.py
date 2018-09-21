# -*- coding: utf-8 -*-
"""Super class of learning models and main entry to run ML
"""
import os
import shutil

class UWOLearningModel:
    """This class is the supoer class of learning models.

    Args:
        traning_data_dir (str): Full directory path of input data images.
                                The files in the directory have following name format.
                                {Label}[_xxxxxx].png
        out_model_dir (str): Full directory path which the learning result is saved to.
        out_label_path (str): Full file path of the label file to write
    """
    def __init__(self, traning_data_dir, out_model_dir, out_label_path):
        self.data_dir = traning_data_dir
        self.model_dir = out_model_dir
        self.label_path = out_label_path

    def learn(self):
        """Start machine learning.

        Run machine running and writing results
        """
        self.clean_output()

        self.run_machine_learning()
        self.__write_label_file(self.get_labels(), self.label_path)

    def clean_output(self):
        """Clean output directories.

        It's inconvenient whenever running. So cleanint output before running.
        """
        self.__clean_output(self.model_dir)
        self.__clean_output(self.label_path)

    def __clean_output(self, path):
        if os.path.exists(path):
            backup = path + ".bak"
            if os.path.isdir(backup):
                shutil.rmtree(backup)
            elif os.path.exists(backup):
                os.remove(backup)
            os.rename(path, path + ".bak")

    def get_full_paths(self):
        """Get Full path list of each input files

        Returns:
            list: Sorted full path list
        """
        input_files = os.listdir(self.data_dir)
        return sorted(map(lambda x: self.data_dir + "/" + x, input_files))

    def get_label_from_path(self, path):
        """Get label from full path
        """
        return os.path.basename(path).split("_")[0]

    def run_machine_learning(self):
        """Run machine learning.
        """
        raise NotImplementedError()

    def one_hot(self, index, depth):
        """Make one-hot format

        Returns:
            list: one-hot encoding format
        """
        ont_hot = [0] * depth
        ont_hot[index] = 1
        return ont_hot

    def maxarg(self, one_hot):
        """Return the index of 1 in ont-hot format

        Returns:
            int: index
        """
        index = 0
        for x in one_hot:
            if x == 1:
                break
            index += 1
        return index

    def get_labels(self):
        """The label list may be different to the input data.

        So, get_labels funtion has to be implemented at each subclasses.

        Returns:
            list: Sorted label list
        """
        raise NotImplementedError()

    def __write_label_file(self, labels, output):
        try:
            os.makedirs(os.path.dirname(output))
        except OSError:
            pass

        with open(output, "w") as file:
            for label in labels:
                file.write(label + "\n")
