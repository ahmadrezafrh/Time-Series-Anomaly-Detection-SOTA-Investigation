### ECOD and COPod 
For training just run the algorithems folder

### MTAD-GAT
For training
- Kuka_v1 dataset must be located in a folder called `our_data/`
- Training Kuka_v1 dataset for 10 epochs, using a lookback (window size) of 150 and a batch size of 32:
```bash 
python train.py --dataset Kuka_v1  --lookback 150 --epochs 10 --bs 32
```