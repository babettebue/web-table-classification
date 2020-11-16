# re-train random forest on new dataset
# use same val split


import pandas as pd
import pickle
import statistics
import sklearn.metrics as skm
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from itertools import compress
from sklearn.model_selection import PredefinedSplit
from joblib import dump, load

pd.set_option("display.max.columns", None)

#load manual features, generated with the DWTC-extractor
pred = pd.read_csv(
    r'E:\Babette\MasterThesis\Classifier_Dresden\predictions2-with-features.csv', header=None)

#define column names
pred = pred.rename(columns={0: "id",
                            1: "y_pred_old_classifier",
                            2: "LOCAL_RATIO_IS_NUMBER_COL_1",
                            3: "LOCAL_RATIO_ANCHOR_ROW_1",
                            4: "RATIO_IMG",
                            5: "LOCAL_RATIO_ANCHOR_COL_1",
                            6: "LOCAL_LENGTH_VARIANCE_COL_1",
                            7: "LOCAL_RATIO_IMAGE_COL_1",
                            8: "LOCAL_RATIO_IMAGE_COL_0",
                            9: "LOCAL_SPAN_RATIO_COL_2",
                            10: "LOCAL_SPAN_RATIO_COL_1",
                            11: "LOCAL_AVG_LENGTH_ROW_2",
                            12: "LOCAL_RATIO_HEADER_ROW_0",
                            13: "RATIO_DIGIT",
                            14: "LOCAL_RATIO_IMAGE_ROW_0",
                            15: "RATIO_ALPHABETICAL",
                            16: "LOCAL_RATIO_IMAGE_ROW_1",
                            17: "LOCAL_RATIO_INPUT_COL_1",
                            18: "LOCAL_RATIO_INPUT_COL_0",
                            19: "LOCAL_RATIO_CONTAINS_NUMBER_ROW_2",
                            20: "LOCAL_AVG_LENGTH_COL_0",
                            21: "RATIO_EMPTY",
                            22: "AVG_ROWS",
                            23: "LOCAL_RATIO_INPUT_ROW_1",
                            24: "LOCAL_RATIO_CONTAINS_NUMBER_COL_2",
                            25: "LOCAL_RATIO_HEADER_COL_1",
                            26: "LOCAL_RATIO_INPUT_ROW_0",
                            27: "AVG_COLS"
                            })
# pred.to_pickle(r'Data\predictions2-with-features.pkl')

# without filtered tables
df_train = pred.copy()
df_train['y_pred_old_classifier'].value_counts()
df_train = df_train[(df_train['y_pred_old_classifier'] != 'OTHER') & (
    df_train['y_pred_old_classifier'] != 'Failure')]
len(df_train)

# define test and train set
ids = np.load(r'E:\Babette\MasterThesis\Classifier_Dresden\all_test_ids.npy')
test_set = []
for index, row in df_train.iterrows():
    if row['id'] in ids:
        test_set.append(True)
    else:
        test_set.append(False)
pred_test = pred[test_set]

x_test = df_train[test_set]
len(x_test)
x_train = df_train[~np.array(test_set)]
len(x_train)


# get real labels
df = pd.read_pickle(r'E:\Babette\MasterThesis\gs_125_warc_files_comb.pkl')
y_train = []
y_train_id = []

for index, row in df.iterrows():
    if row['id'] in x_train['id']:
        y_train_id.append(row['id'])
        if row['label'] == 'genuine':
            y_train.append(1)
        else:
            y_train.append(0)

len(y_train)
len(y_train_id)

y_test = []
y_test_id = []

for index, row in df.iterrows():
    if row['id'] in x_test['id']:
        y_test_id.append(row['id'])
        if row['label'] == 'genuine':
            y_test.append(1)
        else:
            y_test.append(0)

len(y_test)
len(y_test_id)


# define test val split:
v_ids = np.load(r'E:\Babette\MasterThesis\Classifier_Dresden\all_val_ids.npy')
split = []

for index, row in x_train.iterrows():
    if row['id'] in v_ids:
        split.append(0)
    else:
        split.append(-1)


#define pre-defined split
ps = PredefinedSplit(split)


# drop id old classifier prediction and col28
x_train = x_train.drop(['id', 'y_pred_old_classifier', 28], axis=1)
x_test = x_test.drop(['id', 'y_pred_old_classifier', 28], axis=1)


# define classes
n_classes = 2
target_names = ['Layout', 'Genuine']
#--------------------------------------------------------------------------------------------------------------------------------#
# train random forest classifier
#--------------------------------------------------------------------------------------------------------------------------------#

# grid search on val.set
#{'n_estimators': 1600, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 80, 'bootstrap': True}
model = RandomForestClassifier(n_estimators=1600,
                               min_samples_leaf=4,
                               min_samples_split=2,
                               max_depth=80)


model.fit(x_train, y_train)

#load saved model
model = load(r'E:\Babette\MasterThesis\Classifier_Dresden\heuristic_rf.joblib')

y_pred = model.predict(x_test)

# Results RF model Prediction
print('Evaluation on test set:')
print('Acurracy' + str(accuracy_score(y_test, y_pred)))
print(classification_report(y_test, y_pred, target_names=target_names, digits=4))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

# save model
#dump(model, r'E:\Babette\MasterThesis\Classifier_Dresden\heuristic_rf.joblib')
#model= load( r'E:\Babette\MasterThesis\Classifier_Dresden\heuristic_rf.joblib')


# new:

# Acurracy0.9253807106598985
#               precision    recall  f1-score   support

#       Layout       0.92      0.85      0.88       647
#      Genuine       0.93      0.96      0.95      1323

#    micro avg       0.93      0.93      0.93      1970
#    macro avg       0.92      0.91      0.91      1970
# weighted avg       0.93      0.93      0.92      1970

#       Layout     0.9153    0.8516    0.8823       647
#      Genuine     0.9298    0.9615    0.9454      1323

#    micro avg     0.9254    0.9254    0.9254      1970
#    macro avg     0.9226    0.9065    0.9138      1970
# weighted avg     0.9250    0.9254    0.9247      1970

# >>> print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
# [[ 551   96]
#  [  51 1272]]


# Print the feature ranking
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature ranking:")

# for f in range(df_train.shape[1]):
for f in range(11):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    print(df_train.columns[indices[f]])


#--------------------------------------------------------------------------------------------------------------------------------------------#
# default layout classification for filtered out tables
#--------------------------------------------------------------------------------------------------------------------------------------------#
# expand y_pred, and y_test

prediction = pd.DataFrame({'id': y_test_id, 'y_pred': y_pred})


y_test_full = []
y_test_id_full = []

for index, row in df.iterrows():
    if row['id'] in ids:
        y_test_id_full.append(row['id'])
        if row['label'] == 'genuine':
            y_test_full.append(1)
        else:
            y_test_full.append(0)

len(y_test_full)
len(y_test_id_full)

test = pd.DataFrame({'id': y_test_id_full, 'y_test': y_test_full})

ld = test.merge(prediction, on='id', how='left')
ld['y_pred'] = ld['y_pred'].fillna(0)


# Results of RF model Prediction
print('Evaluation on test set:')
print('Acurracy' + str(accuracy_score(ld['y_test'], ld['y_pred'])))
print(classification_report(
    ld['y_test'], ld['y_pred'], target_names=target_names, digits=4))
print(confusion_matrix(ld['y_test'], ld['y_pred'], labels=range(n_classes)))


# Acurracy0.9317315026697178


#           precision    recall  f1-score   support

#       Layout     0.9338    0.9242    0.9290      1267
#      Genuine     0.9298    0.9387    0.9343      1355

#    micro avg     0.9317    0.9317    0.9317      2622
#    macro avg     0.9318    0.9315    0.9316      2622
# weighted avg     0.9318    0.9317    0.9317      2622

# [[1171   96]
#  [  83 1272]]
# alt:
#       Layout       0.91      0.93      0.92      1267
#      Genuine       0.94      0.92      0.93      1355

#    micro avg       0.93      0.93      0.93      2622
#    macro avg       0.93      0.93      0.93      2622
# weighted avg       0.93      0.93      0.93      2622

# [[1181   86]
#  [ 110 1245]]
# Acurracy0.9252479023646072

y_pred = ld['y_pred']
y_test = ld['y_test']
y_test_id = ld['id']
false = []
for i in range(len(y_pred)):
    if y_pred[i] != y_test[i]:
        false.append(y_test_id[i])
len(false)
#np.save(r'E:\Babette\MasterThesis\Feature_extractor\heur_false_id.npy', false)

# hyperparameter tuning

#################################################################################################################################
# cross validation for parameter tuning
#################################################################################################################################

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(random_grid)

rf = RandomForestClassifier()

rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                               n_iter=100, cv=ps, verbose=2, random_state=42, n_jobs=-1)
rf_random.fit(x_train, y_train)
rf_random.best_params_

