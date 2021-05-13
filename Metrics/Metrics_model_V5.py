#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 12:03:31 2019

@author: sebastian
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import f1_score, recall_score,precision_score, accuracy_score
#from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import pandas as pd
#from Read_csv import read_testset, read_trainset
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import coverage_error

#df = read_trainset("/home/sebastian/RNSA/stage_2_train.csv")

test_df = pd.read_csv("./resultados/Resultados_hard/test_hard-1.csv", index_col=0)

pred = pd.read_csv('./resultados/Resultados_hard/prediction_hard-1.csv', index_col=0,  decimal = '.')
#test_df = test_df.iloc[2:]

idx_pred = pred.index
idx_df = test_df.index
diff = idx_pred.difference(idx_df)
diff = np.asarray(diff)

pred = pred.drop(diff, axis=0)
test_df = test_df.sort_index()
pred = pred.sort_index()

y_pred = np.asarray(pred, dtype = np.float64)
y_test = np.asarray(test_df, dtype = np.float64)

ce = coverage_error(y_test,y_pred)
Accuracy =[]
Recall = []
F1_score =[]
for e in range (0,3):
    
    y_test_sample = y_test[:,e] 
    y_pred_sample = np.around(y_pred[:,e])
    
    Accuracy.append(accuracy_score(y_test_sample, y_pred_sample))
    Recall.append(recall_score(y_test_sample, y_pred_sample))
    F1_score.append(f1_score(y_test_sample, y_pred_sample))

metrics = [Accuracy,Recall,F1_score]
#metrics_df = pd.DataFrame(metrics, columns=['Any','Epidural','Intraparenchymal','Intraventricular','Subarachnoid','Subdural'],
#                          index= ['Accuracy','Recall','F1_score'])

#metrics_df.to_csv('resultados/figuras-tablas/metrics-3-no-prepro.csv')



#%% ROC CURVE

n_classes = y_test.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)
lw = 2
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
labels = ['Epidural', 'Subarachnoid','Subdural']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label= labels[i] + ' (area = {0:0.2f})'
             ''.format(roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic to multilabel hemorrhage classification')
plt.legend(loc="lower right")
plt.show()
plt.savefig('resultados/figuras-tablas/ROC_Hard-4.png')
#%%

metric_1 = pd.read_csv("resultados/figuras-tablas/metrics-1-Hard.csv", index_col=0)
metric_2 = pd.read_csv("resultados/figuras-tablas/metrics-2-Hard.csv", index_col=0)
metric_3 = pd.read_csv("resultados/figuras-tablas/metrics-3-Hard.csv", index_col=0)
# metric_4 = pd.read_csv("resultados/figuras-tablas/metrics-4-Cross.csv", index_col=0)
# metric_5 = pd.read_csv("resultados/figuras-tablas/metrics-5-Cross.csv", index_col=0)

df_concat = pd.concat((metric_1, metric_2, metric_3))

by_row_index = df_concat.groupby(df_concat.index)
df_means = by_row_index.mean()
df_means.to_csv('resultados/figuras-tablas/means-hard.csv')

df_std = df_concat.groupby(df_concat.index).agg(np.std, ddof=0)
df_std.to_csv('resultados/figuras-tablas/std-hard.csv')


#%%

def perf_mesure(y_pred,y_test):
    TP = []
    FP =[]
    
    for i in range(len(y_pred)):
        if y_pred[i]==y_test[i]==1:
            TP.append(i)
        if y_test[i]==1 and y_pred[i]!=y_test[i]:
            FP.append(i)
    
    return(TP,FP)

epidural_test = y_test[:,0]
epidural_pred = np.around(y_pred[:,0])

epi_TP, epi_FP = perf_mesure(epidural_pred, epidural_test)

FP_name_epi = idx_df[515]
TP_name_epi = idx_df[966]

suba_test = y_test[:,1]
suba_pred = np.around(y_pred[:,1])

suba_TP, suba_FP = perf_mesure(suba_pred, suba_test)

FP_name_suba = idx_df[11253]
TP_name_suba = idx_df[13923]