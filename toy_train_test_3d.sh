#!/bin/sh

rm -rf toy_data_3d toy_weights_3d toy_images_3d toy_predict_3d
python generate_toy_data_3d.py
python train.py with configs/toy_config_3d.json
python train.py with configs/toy_config_3d.json "bayesian"=False
python test.py with configs/toy_config_3d.json
python test.py with configs/toy_config_3d.json "bayesian"=False

