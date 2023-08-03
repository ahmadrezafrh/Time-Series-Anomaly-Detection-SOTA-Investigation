from torch.backends import cudnn
from modules.preprocessing import load_data
from modules.postprocessing import evaluation
from modules.postprocessing import calculate_threshold
from modules.postprocessing import confusion_rates
from modules.utils import ignore_warnings
from modules.utils import check_path
from solver import Solver

import matplotlib
import numpy as np



ROOTDIR_DATASET_NORMAL = './data/normal'
ROOTDIR_DATASET_ANOMALY = './data/collisions'
cudnn.benchmark = True
ignore_warnings()
select_signals = None
rand_size = None
select_action = None

params = [(1,5, 1),(1,10, 1), (1,15, 1), (1,15, 1), (1,20, 1), (1,25, 1), (1,30, 1),
          (10,20, 1), (10,25, 1), (10,30, 1), (10,35, 1), (10,40, 1), (10,45, 1), (10,50, 1), (10,55, 1), (10,60, 1), (10,65, 1), (10,70, 1), (10,75, 1), (10,80, 1),
          (100,100, 25),(100,105, 25),(100,105, 25),(100,110, 25),(100,115, 25),(100,120, 25),(100,125, 25),(100,130, 25),(100,135, 25),(100,140, 25), (100,145, 25)]

for c, param in enumerate(params):
    
    print(f'training model {c}/{len(params)}')
    print(f'freq = {param[0]}')
    print(f'win_size = {param[1]}')
    print(f'step = {param[2]}')
    
    freq = param[0]
    win_size = param[1]
    step = param[2]
    
    train_config = {'lr': 0.0001,
      'num_epochs': 3,
      'k': 1, 
      'win_size': win_size,
      'input_c': 52,
      'output_c': 52,
      'batch_size': 16,
      'pretrained_model': '20',
      'dataset': 'KUKA',
      'mode': 'train',
      'data_path': 'dataset',
      'model_save_path':'checkpoints',
      'anormly_ratio': 1.0,
      'freq': freq,
      'step': step}
    

    loader = load_data(freq=freq, rand_size=rand_size, select_signals=select_signals, select_action=select_action)
    train_data, df = loader.train(ROOTDIR_DATASET_NORMAL)
    del loader
    
    check_path(train_config['model_save_path'])
    solver_train = Solver(train_config)
    solver_train.train()
    print('\n\n\n\n')

    
    loader = load_data(freq=freq, rand_size=rand_size, select_signals=select_signals, select_action=select_action)
    test_data, labels, df = loader.test(ROOTDIR_DATASET_ANOMALY)
    
    test_config = {'lr': 0.0001,
      'num_epochs': 10,
      'k': 1, 
      'win_size': win_size,
      'input_c': 52,
      'output_c': 52,
      'batch_size': 16,
      'pretrained_model': '20',
      'dataset': 'KUKA',
      'mode': 'test',
      'data_path': 'dataset',
      'model_save_path':'checkpoints',
      'anormly_ratio': 1.0,
      'freq': freq,
      'step': 1}  
    
    solver_test = Solver(test_config)
    test = solver_test.test(freq, win_size)
    np.save(f'./results/test_{freq}_{win_size}.npy', test)
    np.save(f'./results/labels_{freq}_{win_size}.npy', labels)



results = {}
ars = [1,2,3,4,5,6,7,8,9,10]
for c, param in enumerate(params):
    
    print(f'evaluating model {c}/{len(params)}')
    print(f'freq = {param[0]}')
    print(f'win_size = {param[1]}')
    freq = param[0]
    win_size = param[1]
    results[f'{freq}_{win_size}'] = {}

    
    
    
    test = np.load(f'./results/test_{freq}_{win_size}.npy', allow_pickle=True)
    labels = np.load(f'./results/labels_{freq}_{win_size}.npy', allow_pickle=True)
    
    
    
    for anormly_ratio in ars:
        print(f'anomaly ratio: {anormly_ratio}')
        threshold = calculate_threshold(test, anormly_ratio)
        print(f'threshold : {threshold}\n')
        pred, result = evaluation(test, labels, threshold=threshold, win=win_size, log=False)
        results[f'{freq}_{win_size}'][anormly_ratio] = result
    
    
    print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')


# ----------------------------------

import matplotlib.pyplot as plt
import numpy as np

ars = [4, 5, 6]
freq = 100
win_sizes = []
for i in range(10):
    win_sizes.append(100 + i*5)
    
    

for ar in ars:
    y = []
    for ws in win_sizes:
        y.append(results[f'{freq}_{ws}'][ar][2])
    
    plt.plot(win_sizes, y)
    plt.show()
        

    

# ------------------------------


ar = 6
freq = 100
win_sizes = []
for i in range(10):
    win_sizes.append(100+i*5)

    
for ws in win_sizes:
    print(f'window_size: {ws}')
    print(f"acc = {results[f'{freq}_{ws}'][ar][0]}\nrecall = {results[f'{freq}_{ws}'][ar][1]}\nspec = {results[f'{freq}_{ws}'][ar][2]}\nprec = {results[f'{freq}_{ws}'][ar][3]}\nf1= {results[f'{freq}_{ws}'][ar][4]}\nfb= {results[f'{freq}_{ws}'][ar][5]}")
    print('\n')
    
    
    
    
# ------------------------------

freq = 1
win_size = 10
test = np.load(f'./results/test_{freq}_{win_size}.npy', allow_pickle=True)
labels = np.load(f'./results/labels_{freq}_{win_size}.npy', allow_pickle=True)
fpr0, fnr0, tpr0= confusion_rates(test,  labels, win_size, num=80)


freq = 10
win_size = 50
test = np.load(f'./results/test_{freq}_{win_size}.npy', allow_pickle=True)
labels = np.load(f'./results/labels_{freq}_{win_size}.npy', allow_pickle=True)
fpr1, fnr1, tpr1 = confusion_rates(test,  labels, win_size, num=80)


freq = 100
win_size = 130
test = np.load(f'./results/test_{freq}_{win_size}.npy', allow_pickle=True)
labels = np.load(f'./results/labels_{freq}_{win_size}.npy', allow_pickle=True)
fpr2, fnr2, tpr2 = confusion_rates(test,  labels, win_size, num=80)



plt.figure(figsize=(7, 7), dpi=100)
plt.plot(fpr0,fnr0, color='darkblue', label='Frequency=1, Win_size=10')
plt.plot(fpr1,fnr1, color='darkred', label='Frequency=10, Win_size=50')
plt.plot(fpr2,fnr2, color='darkorange', label='Frequency=100, Win_size=130')
plt.title('DET plot')
ax = plt.gca()
ax.set_yscale('log')
ax.set_xscale('log')
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_xaxis().set_tick_params(which='minor', size=0)
ax.get_xaxis().set_tick_params(which='minor', width=0) 
plt.xlim([0.2, 100])
plt.ylim([0, 105])
plt.xlabel('False Positive Rate')
plt.ylabel("False Negative Rate")
plt.grid(linestyle='--')
plt.legend()
plt.tight_layout()
plt.savefig('./results/det.png')



# ------------------------------



plt.figure(figsize=(7, 7), dpi=100)
plt.plot(fpr0,tpr0, color='darkblue', label='Frequency=1, Win_size=10')
plt.plot(fpr1,tpr1, color='darkred', label='Frequency=10, Win_size=50')
plt.plot(fpr2,tpr2, color='darkorange', label='Frequency=100, Win_size=130')
plt.title('DET plot')
ax = plt.gca()
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.get_xaxis().set_tick_params(which='minor', size=0)
ax.get_xaxis().set_tick_params(which='minor', width=0) 
plt.xlabel('False Positive Rate')
plt.ylabel("True Positive Rate")
plt.grid(linestyle='--')
plt.legend()
plt.tight_layout()
plt.savefig('./results/roc.png')





