# Installation guide

git clone https://github.com/rhllasag/fedLabSync.git

# Conda Environment

conda env create --file environment.yaml

conda activate fedLabSync

pip install fedLabSync

pip install psg_utils==0.1.6

pip install typing-extensions==4.6.0

pip install tables

pip install tables

pip install carbontracker

pip install tensorflow_addons

# Pypi Installation in case of fedLabSync unavailability

python setup.py bdist_wheel

pip install dist/fedLabSync-0.0.3-py3-none-any.whl --force-reinstall

# Hybrid Federated Learning

cd /utime/bin

fd preprocessing --dataset_path /data/cmapss --elbow_point 120 --FD00x 4 --operating_regimes 3 --out_path /data/cmapss/processed

fd splitting --dataset_path /data/cmapss/processed/ --FD00x 4 --nodes 4 --model utime --seed 5

fd running --dataset_path /data/cmapss/processed/ --nodes 1 --model utime --input_signals 16

# U-Time

cd utime_model

fd initializing --name utime_model --model utime --data_dir /data/cmapss/processed/data-centralized

fd training --dataset_path \data\cmapss\processed\data-centralized-utime\ --num_gpus 1 --overwrite
