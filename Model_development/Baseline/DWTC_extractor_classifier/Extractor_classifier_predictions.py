# evaluate dresden classifier predictions 

import pandas as pd
import numpy as np
import pickle
from bs4 import BeautifulSoup
import statistics
import sklearn.metrics as skm
from itertools import compress
from sklearn.metrics import accuracy_score

#load predictions generated with the DWTC extractor Weka Random Forest classifier
pred = pd.read_csv(r'predictions2-with-subtables-filtered.csv', header=None)

#clean csv
pred = pred.rename(columns={0: "id", 1: "y_pred"})
pred.loc[pred['y_pred'] == 'RELATIONf', 'y_pred'] = 'RELATION'
pred.loc[pred['y_pred'] == 'OTHER', 'y_pred'] = 'Nested'


pred_n = pred[(pred['y_pred'] != 'Nested') & (pred['y_pred'] != 'Failure')]

predict = []
predict_id = []

for index, row in pred.iterrows():
    if row['y_pred'] == 'RELATION':
        predict.append(1)
        predict_id.append(row['id'])
    elif row['y_pred'] == 'LAYOUT':
        predict.append(0)
        predict_id.append(row['id'])
#    elif row['y_pred']== 'Nested':
#        predict.append(0)
#        predict_id.append(row['id'])
    else:
        continue

len(predict)
len(predict_id)

# load gold standard labels
df = pd.read_pickle(r'E:\Babette\MasterThesis\gs_125_warc_files_comb.pkl')

label = []
label_id = []

for index, row in df.iterrows():
    if row['id'] in predict_id:
        label_id.append(row['id'])
        if row['label'] == 'genuine':
            label.append(1)
        else:
            label.append(0)

len(label)
len(label_id)


# subset for test set
ids = np.load(r'E:\Babette\MasterThesis\Classifier_Dresden\all_test_ids.npy')

test_set = []
for i in predict_id:
    #    if i in train_test['id']:
    if i in ids:
        test_set.append(True)
    else:
        test_set.append(False)

predict = list(compress(predict, test_set))
label = list(compress(label, test_set))

report = skm.classification_report(label, predict, digits=4)
print(report)

# confusion Matrix
cm = skm.confusion_matrix(label, predict)
print(cm)

print('Acurracy' + str(accuracy_score(predict, label)))

##############  Default prediction Layout  for Nested tables ######################################################################################


ids = np.load(r'E:\Babette\MasterThesis\Classifier_Dresden\all_test_ids.npy')
test_set = []
for index, row in pred.iterrows():
    if row['id'] in ids:
        test_set.append(True)
    else:
        test_set.append(False)
pred_test = pred[test_set]

len(pred_test)

predict = []
predict_id = []

for index, row in pred_test.iterrows():
    if row['y_pred'] == 'RELATION':
        predict.append(1)
        predict_id.append(row['id'])
    elif row['y_pred'] == 'LAYOUT':
        predict.append(0)
        predict_id.append(row['id'])
    elif row['y_pred'] == 'Nested':
        predict.append(0)
        predict_id.append(row['id'])
    elif row['y_pred'] == 'Failure':
        predict.append(0)
        predict_id.append(row['id'])
    else:
        continue

len(predict)
len(predict_id)

#df = pd.read_pickle(r'E:\Babette\MasterThesis\gs_125_warc_files_comb.pkl')

label = []
label_id = []

for index, row in df.iterrows():
    if row['id'] in predict_id:
        label_id.append(row['id'])
        if row['label'] == 'genuine':
            label.append(1)
        else:
            label.append(0)

len(label)
len(label_id)


#Evaluation
report = skm.classification_report(label, predict, digits=4)
print(report)
# confusion Matrix
cm = skm.confusion_matrix(label, predict)
print(cm)
print('Acurracy' + str(accuracy_score(predict, label)))

# [[1161  106]
#  [ 183 1172]]

#               precision    recall  f1-score   support

#            0     0.8638    0.9163    0.8893      1267
#            1     0.9171    0.8649    0.8902      1355

#    micro avg     0.8898    0.8898    0.8898      2622
#    macro avg     0.8904    0.8906    0.8898      2622
# weighted avg     0.8913    0.8898    0.8898      2622

# Acurracy0.8897787948131197
