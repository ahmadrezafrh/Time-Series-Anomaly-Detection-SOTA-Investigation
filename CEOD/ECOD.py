from darts import TimeSeries
from darts.models import NaiveSeasonal, NBEATSModel
from pyod.models.ecod import ECOD
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, fbeta_score
from torch.backends import cudnn
import numpy as np
from modules.preprocessing import load_data
from modules.postprocessing import evaluation
from modules.utils import ignore_warnings

from pyod.models.copod import COPOD

ROOTDIR_DATASET_NORMAL = './data/normal'
ROOTDIR_DATASET_ANOMALY = './data/collisions'
cudnn.benchmark = True
ignore_warnings()

select_signals = None
rand_size = None
select_action = None
freq = 10

train_config = {'lr': 0.0001,
                'num_epochs': 3,
                'k': 1,
                'win_size': 70,
                'input_c': 57,
                'output_c': 57,
                'batch_size': 32,
                'pretrained_model': '20',
                'dataset': 'KUKA',
                'mode': 'train',
                'data_path': 'dataset',
                'model_save_path': 'checkpoints',
                'anormly_ratio': 1.0}

test_config = {'lr': 0.0001,
               'num_epochs': 10,
               'k': 1,
               'win_size': 70,
               'input_c': 57,
               'output_c': 57,
               'batch_size': 32,
               'pretrained_model': '20',
               'dataset': 'KUKA',
               'mode': 'test',
               'data_path': 'dataset',
               'model_save_path': 'checkpoints',
               'anormly_ratio': 1.0}

dataFrequency = [1,10, 100, 200]
outlierFreq = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
for j in dataFrequency:
    loader = load_data(freq=j, rand_size=rand_size, select_signals=select_signals, select_action=select_action)
    train_data, df = loader.train(ROOTDIR_DATASET_NORMAL)
    test_data, labels, df = loader.test(ROOTDIR_DATASET_ANOMALY)
    val_data, vallabels, df = loader.test(ROOTDIR_DATASET_ANOMALY)
    for i in outlierFreq:
        model = ECOD(i)
        model.fit(train_data)
        pred = model.predict(val_data)
        gt = vallabels
        tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()
        specificity = tn / (tn + fp)
        Sensitivity = tp / (tp + fn)
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        fb_score = fbeta_score(gt, pred, average='macro', beta=2)
        print(f' model:COPOD ,dataFrequency:{j} ,OutlierFrequency: {i:.4f},window size:{0}')
        print(
            " Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, fBata-score : {:0.4f}, specificity : {:0.4f}, Sensitivity:{:0.4f}".format(
                accuracy, precision,
                recall, f_score, fb_score, specificity, Sensitivity))
        windowsize = []
        if j == 1:
            windowsize = [5, 15, 20, 25, 30, 35, 40, 45]
        elif j == 10:
            windowsize = [20, 45, 50, 55, 60, 65, 70, 75, 80]
        elif j == 100:
            windowsize = [105, 110, 115, 120, 125, 130, 135, 140, 145]
        else:
            windowsize = [100, 200, 400, 600, 800, 1000]
        prediction = model.decision_function(val_data)
        for r in windowsize:
            meanPrediction = []
            for z in range(prediction.size - (prediction.size % r)):
                meanPrediction.append(np.mean(prediction[z:z + r]))
            print(f' model:ECOD ,dataFrequency:{j} ,OutlierFrequency: {i:.4f} , window size:{r} ')
            hi = vallabels[:len(meanPrediction)]
            evaluation(meanPrediction, vallabels[:len(meanPrediction)], threshold=model.threshold_, win=r, log=False)
