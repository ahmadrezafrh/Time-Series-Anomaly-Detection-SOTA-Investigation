export CUDA_VISIBLE_DEVICES=0

python3 main.py --anormly_ratio 6 --win_size 130 --num_epochs 3   --batch_size 32  --mode train --dataset KUKA  --data_path dataset  --k 1  --input_c 52 --output_c 52 --freq 100 --step 25
python3 main.py --anormly_ratio 6 --win_size 60 --num_epochs 10   --batch_size 32  --mode test    --dataset KUKA   --data_path dataset  --k 1   --input_c 52 --output_c 52 --freq 100 --step 25

