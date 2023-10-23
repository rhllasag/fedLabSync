# Pypi
python setup.py bdist_wheel

pip install  dist/fedLabSync-0.0.3-py3-none-any.whl  --force-reinstall

# Hybrid Federated Learning 

cd /utime/bin

fd preprocessing --dataset_path /data/cmapss --elbow_point 120 --FD00x 4 --operating_regimes 3  --out_path /data/cmapss/processed

fd splitting --dataset_path /data/cmapss/processed/ --FD00x 4  --nodes 1 --model cnn

fd running --dataset_path /data/cmapss/processed/ --nodes 1 --model cnn


# U-Time

fd initializing --name utime_model --model utime --data_dir /data/cmapss/processed/data-centralized