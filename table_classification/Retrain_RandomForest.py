# re-train random forest on new dataset


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

pd.set_option("display.max.columns", None)
pred= pd.read_csv(r'E:\Babette\MasterThesis\Classifier_Dresden\predictions2-with-features.csv', header= None)

pred= pred.rename(columns={0: "id",
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

pred[28].describe()
df_train = pred.copy()
df_train['y_pred_old_classifier'].value_counts()

#select tables with in dwtc-extractor created features for now
df_train= df_train[(df_train['y_pred_old_classifier']!= "OTHER") & (df_train['y_pred_old_classifier']!= "Failure")]
#drop old classifier prediction and col28
df_train= df_train.drop(['y_pred_old_classifier', 28], axis=1)
df_train.columns
#get real labels

df = pd.read_pickle(r'E:\Babette\MasterThesis\gs_125_warc_files_comb.pkl')


label=[]
label_id= []

for index, row in df.iterrows():
    if row['id'] in df_train['id']:
        label_id.append(row['id'])
        if row['label']== 'genuine':
            label.append(1)
        else:
            label.append(0)

len(label)
len(label_id)
len(df_train)


# train random forest classifier

# define the model
model = RandomForestClassifier()
model
#repeated random subsampling : shuffle and split
ss = ShuffleSplit(n_splits=100, test_size=0.10, random_state=1)

accuracy_unpruned = cross_val_score(model, df_train, label, scoring='accuracy', cv=ss, n_jobs=-1, error_score='raise')
accuracy_unpruned

n_scores = cross_val_score(model, df_train, label,  cv=ss, n_jobs=-1, scoring='f1' , error_score='raise')


# report performance
print('Accuracy unpruned: %.3f (%.3f)' % (np.mean(accuracy_unpruned), np.std(accuracy_unpruned)))
#Accuracy unpruned: 0.919 (0.008) (without nested)
print('F1 unpruned: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
#F1 unpruned: 0.940 (0.006) (without nested)


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
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

rf= RandomForestClassifier()

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(df_train, label)
rf_random.best_params_


param_grid = {'n_estimators': [100, 200],
        'max_features': ['auto'],
        'max_depth': [5, 10, 15],
        'min_samples_split': [4, 5, 6],
        'min_samples_leaf': [3, 4, 5],
        'bootstrap': [True]
        }


rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = ss, n_jobs = -1, verbose = 2)

grid_search.fit(df_train, label)
grid_search.best_params_
best_grid = grid_search.best_estimator_


n_scores = cross_val_score(best_grid, df_train, label, scoring='accuracy', cv=ss, n_jobs=-1, error_score='raise')
accuracy_grid_search = n_scores

n_scores = cross_val_score(best_grid, df_train, label,  cv=ss, n_jobs=-1, scoring='f1' , error_score='raise')

# report performance
print('Accuracy: %.3f (%.3f)' % (np.mean(accuracy_grid_search), np.std(accuracy_grid_search)))
print('F1: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

# without nested tables
#{'bootstrap': True, 'max_depth': 15, 'max_features': 'auto', 'min_samples_leaf': 3, 'min_samples_split': 4, 'n_estimators': 200}
#F1: 0.946
#Accuracy: 0.927 

skm.SCORERS.keys()