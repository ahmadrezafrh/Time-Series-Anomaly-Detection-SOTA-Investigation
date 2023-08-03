from modules.utils import infinity
from sklearn.metrics import precision_recall_fscore_support, fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np




def evaluation(test, labels, threshold, win, log=False):
    pred = []
    i = 0
    while i <len(test):
        if 1 in labels[i:i + win]:
            for c in infinity():
                if labels[i + win + c] != 1:
                    break
            for k in range(labels[i:i + win + c].shape[0]):
                if test[i + k] >= threshold:
                    detect = True
                    break
                else:
                    detect = False
            if log:
                print(f'anomaly in {i + win - 1} - {i + win + c - 1} - {detect}')

            if detect:
                for n in range(labels[i:i + win + c].shape[0]):
                    if labels[i + n] == 1:
                        pred.append(1)
                    else:
                        pred.append(0)

            else:
                for n in range(labels[i:i + win + c].shape[0]):
                    if labels[i + n] == 1:
                        pred.append(0)
                    else:
                        pred.append(0)

            i = i + win + c


        else:
            
            if i==len(test)-1:
                for n in range(win):
                    pred.append(0)
                    
                        
            else:
                if test[i] >= threshold:
                    pred.append(1)
                else:
                    pred.append(0)
                    
            i+=1
    
    
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
    conf = [tn, fp, fn, tp]
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(labels, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(labels, pred,
                                                                          average='binary')
    fb_score = fbeta_score(labels, pred, average='macro', beta=2)
    results = []
    if log:
        print('====================================')
        print(
            "Accuracy : {:0.4f}\nPrecision : {:0.4f}\nRecall : {:0.4f}\nF-score : {:0.4f}\nfBata-score : {:0.4f}\nspecificity : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score, fb_score, specificity))
    
    results.append(accuracy)
    results.append(recall)
    results.append(specificity)
    results.append(precision)
    results.append(f_score)
    results.append(fb_score)
    
    
    return pred, results, conf



def confusion_rates(test, labels, win, num=80):
    
    min_th = np.log(min(test))
    max_th = np.log(max(test))
    thresholds = 10**np.linspace(min_th, max_th, num=num)

    
    
    fpr = []
    fnr = []
    tpr = []
    for th in thresholds:
        _, _, conf = evaluation(test, labels,th, win)

        tn = conf[0]
        fp = conf[1]
        fn = conf[2]
        tp = conf[3]
    
        
        fpr.append(100*fp/(fp+tn))
        fnr.append(100*fn/(fn+tp))
        tpr.append(100*tp/(tp+fn))
    
    return fpr, fnr, tpr



def calculate_threshold(combined_energy, anormly_ratio):
    return np.percentile(combined_energy, 100 - anormly_ratio)