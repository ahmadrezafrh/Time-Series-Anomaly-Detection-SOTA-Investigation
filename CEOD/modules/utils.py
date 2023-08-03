# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import os
import torch
import warnings

from modules.preprocessing import seaborn_cm
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable



def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def infinity():
    index = 0
    while True:
        yield index
        index += 1

def evaluate(classifier, dataset, ground_truth, action2int):
    predictions = classifier.predict(dataset)
    cm = confusion_matrix(ground_truth, predictions, labels=list(action2int.values()))
    fig, ax = plt.subplots(1, 1, figsize=(13, 13))
    int2action = {v: k for k, v in action2int.items()}
    seaborn_cm(cm, ax, [int2action[l] for l in action2int.values()], fontsize=9, xrotation=90)
    
    
    
def plot_correlation_matrix(df, ax, show_ticks=False):
    mat = ax.matshow(df.corr())
    if show_ticks:
        ax.set_xticks(range(df.select_dtypes(['number']).shape[1]), 
                df.select_dtypes(['number']).columns, 
                fontsize=8, rotation=270)
        ax.set_yticks(range(df.select_dtypes(['number']).shape[1]), 
                df.select_dtypes(['number']).columns, 
                fontsize=8, rotation=0)
    else:
        ax.axis("off")
    ax.grid(False)
    cb = ax.colorbar(mat, ax=ax)
    cb.ax.tick_params(labelsize=14, labelrotation=90)
    ax.set_title('Correlation Matrix', fontsize=16);

def ignore_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
def check_path(path):
    if (not os.path.exists(path)):
        os.mkdir(path)