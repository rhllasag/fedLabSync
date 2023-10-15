# Hybrid Federated Learning 

cd /utime/bin

python preprocessing.py --dataset_path /data/cmapss --elbow_point 120 --FD00x 4 --operating_regimes 3  --out_path /data/cmapss/processed

python splitting.py --dataset_path /data/cmapss/processed/ --FD00x 4  --nodes 1 --model cnn

python running.py --dataset_path /data/cmapss/processed/ --nodes 1 --model cnn