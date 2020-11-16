# re-train random forest on new dataset with visual features


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
from joblib import dump, load
from sklearn.model_selection import PredefinedSplit


pd.set_option("display.max.columns", None)

#load manual features & VGG16 features dataset
df_feat = pd.read_pickle(r'E:\Babette\MasterThesis\Feature_extractor\VGG16_finetuned\manual_vgg16_finetuned_features_dataset.pkl')
#df_feat = pd.read_pickle(r'manual_vgg16_finetuned_features_dataset.pkl')


df_train = df_feat[df_feat['set']=='train']
df_test = df_feat[df_feat['set']=='test']
df_train= df_train.drop(['set',], axis=1)
df_test= df_test.drop(['set',], axis=1)


#load GS labels
df = pd.read_pickle(r'E:\Babette\MasterThesis\gs_125_warc_files_comb.pkl')
#df = pd.read_pickle(r'gs_125_warc_files_comb.pkl')
#df = df[filter]

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
#test set
label_test=[]
label_test_id= []

for index, row in df.iterrows():
    if row['id'] in df_test['id']:
        label_test_id.append(row['id'])
        if row['label']== 'genuine':
            label_test.append(1)
        else:
            label_test.append(0)

len(label_test)
len(label_test_id)
len(df_train)



#simple mean imputation
for x in df_test.iloc[:,1:27].columns:
        df_test[x].fillna(df_test[x].mean() , inplace = True)

for x in df_train.iloc[:,1:27].columns:
        df_train[x].fillna(df_train[x].mean() , inplace = True)


# define test val split:
v_ids= np.load(r'E:\Babette\MasterThesis\Classifier_Dresden\all_val_ids.npy')
split=[]
#va=[]
for index, row in df_train.iterrows():
    if row['id'] in v_ids:
        split.append(0)
#        va.append(False)
    else: 
        split.append(-1)
#        va.append(True)


ps = PredefinedSplit(split)

df_test= df_test.drop(['id',], axis=1)
df_train= df_train.drop(['id',], axis=1)

n_classes=2
target_names= ['Layout', 'Genuine']
# train random forest classifier w. default params ###############################################################################

# define the model
model= load( r'E:\Babette\MasterThesis\Feature_extractor\VGG16_finetuned\With_nested_mean_imputation\heuristic_vgg16_rf.joblib') 

#after grid search
#{'n_estimators': 311, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 90, 'bootstrap': False}
#model = RandomForestClassifier(n_estimators= 311,
#                                min_samples_leaf= 1,
#                                min_samples_split= 5,
#                                max_depth=90,
#                                bootstrap= False ) 

#accuracy_unpruned = cross_val_score(model, df_train, label, scoring='accuracy', cv=ss, n_jobs=-1, error_score='raise')
#n_scores = cross_val_score(model, df_train, label,  cv=ss, n_jobs=-1, scoring='f1' , error_score='raise')

model.fit(df_train, label)

y_pred = model.predict(df_test)

# Results of default RF model Prediction
print('Evaluation on test set:')
print('Acurracy' + str(accuracy_score(label_test, y_pred)))
print(classification_report(label_test, y_pred, target_names=target_names, digits=4))
print(confusion_matrix(label_test, y_pred, labels=range(n_classes)))


#save model
#dump(model, r'E:\Babette\MasterThesis\Feature_extractor\VGG16_finetuned\With_nested_mean_imputation\heuristic_vgg16_rf.joblib') 
#model= load( r'E:\Babette\MasterThesis\Feature_extractor\VGG16_finetuned\With_nested_mean_imputation\heuristic_vgg16_rf.joblib') 


# Print the feature ranking
importances = model.feature_importances_
#std = np.std([best_grid.feature_importances_ for tree in best_grid.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")

#for f in range(df_train.shape[1]):
for f in range(11):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    print(df_train.columns[indices[f]])

#save misclassified ids
false=[]
for i in range(len(y_pred)):
    if y_pred[i]!= label_test[i]:
        false.append(label_test_id[i])
len(false)

#np.save(r'E:\Babette\MasterThesis\Feature_extractor\VGG16_finetuned\With_nested_mean_imputation\vgg16_manual_false_id.npy', false)


# train random forest classifier with grid search param optimization ###################################################################

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
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
#repeated random subsampling : shuffle and split
#ss = ShuffleSplit(n_splits=100, test_size=0.10, random_state=1)

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = ps, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(df_train, label)

print('Best parameter setting found:')
print(rf_random.best_params_)
best_grid = rf_random.best_estimator_


y_pred = best_grid.predict(df_test)

# Results of default RF model Prediction
print('Evaluation on test set:')
print('Acurracy' + str(accuracy_score(label_test, y_pred)))
print(classification_report(label_test, y_pred, target_names=target_names))
print(confusion_matrix(label_test, y_pred, labels=range(n_classes)))



#----------------------------------------------------------------------------------------------------------------------#
#Only visual features
#----------------------------------------------------------------------------------------------------------------------#
#only use visual features
df_train_vis=df_train.iloc[:,26:538]
df_test_vis=df_test.iloc[:,26:538]

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)]
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

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = ps, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(df_train_vis, label)

print('Best parameter setting found:')
print(rf_random.best_params_)
best_grid = rf_random.best_estimator_


y_pred = best_grid.predict(df_test_vis)


# Results of  RF model Prediction
print('Evaluation on test set:')
print('Acurracy' + str(accuracy_score(label_test, y_pred)))
print(classification_report(label_test, y_pred, target_names=target_names, digits=4))
print(confusion_matrix(label_test, y_pred, labels=range(n_classes)))


#dump(best_grid, r'E:\Babette\MasterThesis\Feature_extractor\VGG16_finetuned\With_nested_mean_imputation\vgg16_rf.joblib') 
best_grid= load( r'E:\Babette\MasterThesis\Feature_extractor\VGG16_finetuned\With_nested_mean_imputation\vgg16_rf.joblib') 

importances = best_grid.feature_importances_
std = np.std([best_grid.feature_importances_ for tree in best_grid.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(0,11):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))


#save misclassified ids
false=[]
for i in range(len(y_pred)):
    if y_pred[i]!= label_test[i]:
        false.append(label_test_id[i])
len(false)

#np.save(r'E:\Babette\MasterThesis\Feature_extractor\VGG16_finetuned\With_nested_mean_imputation\vgg16_false_id.npy', false)

print('Script finished successfully')