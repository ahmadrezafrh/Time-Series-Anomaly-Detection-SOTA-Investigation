from modules.utils import infinity
from sklearn.metrics import precision_recall_fscore_support, fbeta_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def evaluation(test, labels, threshold, win, log=True):
    pred = []
    i = 0
    while i < len(test):
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

            if i == len(test) - 1:
                for n in range(win):
                    pred.append(0)


            else:
                if test[i] >= threshold:
                    pred.append(1)
                else:
                    pred.append(0)

            i += 1

    pred = pred[:len(labels)]
    tn, fp, fn, tp = confusion_matrix(labels, pred[:len(labels)]).ravel()
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(labels, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(labels, pred,
                                                                          average='binary')
    fb_score = fbeta_score(labels, pred, average='macro', beta=2)
    Sensitivity = tp / (tp + fn)
    print(
        "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, fBata-score : {:0.4f}, specificity : {:0.4f}, Sensitivity:{:0.4f}".format(
            accuracy, precision,
            recall, f_score, fb_score, specificity, Sensitivity))

    return labels, pred


def evaluation_bes(test, labels, threshold, win, log=True):
    pred = []
    i = 0
    while i < len(test):
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

            if i == len(test) - 1:
                for n in range(win):
                    pred.append(0)


            else:
                if test[i] >= threshold:
                    pred.append(1)
                else:
                    pred.append(0)

            i += 1

    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(labels, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(labels, pred,
                                                                          average='binary')
    fb_score = fbeta_score(labels, pred, average='macro', beta=2)
    print(
        "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, fBata-score : {:0.4f}, specificity : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score, fb_score, specificity))

    return labels, pred


def evaluation_sec(test, labels, threshold, win, log=True):
    pred = []
    gt = []
    i = 0
    while i < len(test):
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
                    if test[i + n] >= threshold:
                        gt.append(1)
                        pred.append(1)
                    else:
                        if labels[i + n] == 0:
                            gt.append(0)
                            pred.append(0)
                        if labels[i + n] == 1:
                            gt.append(1)
                            pred.append(1)
            else:
                for n in range(labels[i:i + win + c].shape[0]):
                    if test[i + n] >= threshold:
                        gt.append(0)
                        pred.append(1)
                    else:
                        if labels[i + n] == 0:
                            gt.append(0)
                            pred.append(0)
                        if labels[i + n] == 1:
                            gt.append(1)
                            pred.append(0)
            i = i + win + c


        else:
            if test[i] >= threshold:
                gt.append(0)
                pred.append(1)
            else:
                gt.append(0)
                pred.append(0)
            i += 1

    tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()
    specificity = tn / (tn + fp)
    Sensitivity = tp / (tp + fn)
    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                          average='binary')
    fb_score = fbeta_score(gt, pred, average='macro', beta=2)
    print(
        "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, fBata-score : {:0.4f}, specificity : {:0.4f}, Sensitivity:{:0.4f}".format(
            accuracy, precision,
            recall, f_score, fb_score, specificity, Sensitivity))
