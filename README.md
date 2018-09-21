=========
uwo_ps_ml
==========

uwo_ps_ml is machine learing code and pre-processing tools for
UWO(Uncharted Waters Online) Price Share Aide

Requirements:
  - pip install tensorflow
  - pip install pillow
  - pip install --upgrade git+https://github.com/ommokazza/uwo_ps_utils.git@master#egg=uwo_ps_utils

Start machine learning
  - python ml.py

Tools for data pre-processing
  - make_training_data.py
    Extracts training data from screenshot and label file.
  - make_label_text.py
    Make label file for a screenshot by using learned model.
