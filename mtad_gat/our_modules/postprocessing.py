from our_modules.utils import infinity
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def evaluation(test, labels, threshold, win, log=True):
    pred = []
    gt = []
    i = 0
    while i<test.shape[0] :
        if 1 in labels[i:i+win]:
            for c in infinity():
                if labels[i+win+c]!=1:
                    break
            for k in range(labels[i:i+win+c].shape[0]):
                if test[i+k]>=threshold:
                    detect = True
                    break
                else:
                    detect = False
            if log:
                print(f'anomaly in {i+win-1} - {i+win+c-1} - {detect}')
            
            if detect:
               for n in range(labels[i:i+win+c].shape[0]):
                   if test[i+n]>=threshold:
                       gt.append(1)
                       pred.append(1)
                   else:
                       if labels[i+n]==0:
                           gt.append(0)
                           pred.append(0)
                       if labels[i+n]==1:
                           gt.append(1)
                           pred.append(1)
            else:
                for n in range(labels[i:i+win+c].shape[0]):
                    if test[i+n]>=threshold:
                        gt.append(0)
                        pred.append(1)
                    else:
                        if labels[i+n]==0:
                            gt.append(0)
                            pred.append(0)
                        if labels[i+n]==1:
                            gt.append(1)
                            pred.append(0)
            i = i+win+c
            
            
        else:
            if test[i]>=threshold:
                gt.append(0)
                pred.append(1)
            else:
                gt.append(0)
                pred.append(0)
            i+=1
                       
    
    
    tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()
    specificity = tn / (tn+fp)
    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                          average='binary')
    print(
        "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, specificity : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score, specificity)) 