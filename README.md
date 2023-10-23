# Pypi
python setup.py bdist_wheel

pip install  dist/fedLabSync-0.0.2-py3-none-any.whl  --force-reinstall

# Hybrid Federated Learning 

cd /utime/bin

python preprocessing.py --dataset_path /data/cmapss --elbow_point 120 --FD00x 4 --operating_regimes 3  --out_path /data/cmapss/processed

python splitting.py --dataset_path /data/cmapss/processed/ --FD00x 4  --nodes 1 --model cnn

python running.py --dataset_path /data/cmapss/processed/ --nodes 1 --model cnn


# U-Time

python initializing.py --name utime_model --model utime --data_dir /data/cmapss/processed/data-centralized