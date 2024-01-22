import os
import pandas as pd
import jnius_config
resources = os.path.join('runtime_testing', 'resources')
jar_file_path = os.path.join(resources, 'dwtc-extension-1.0-jar-with-dependencies.jar')
jnius_config.set_classpath(jar_file_path)

from jnius import autoclass


def calculate_manual_features(dataset_path, random_forest_model_path):
    csv_dataset_path = pickle_to_formatted_csv(dataset_path)
    features_path = os.path.join(os.path.dirname(os.path.realpath(dataset_path)), 'features.csv')

    RF = autoclass('webreduce.extension.classification.TableClassificationUtils')
    rf_classifier = RF(random_forest_model_path)
    rf_classifier.writeFeatures(csv_dataset_path, features_path)

    df_man = pd.read_csv(features_path, header=None)
    df_man.rename({0: 'id'}, axis='columns', inplace=True)
    # feature #27 from manual html based approach is a dummy feature and can be deleted
    df_man.drop(df_man.columns[27], axis=1, inplace=True)

    return df_man


def make_prediction(dataset_path, random_forest_model_path):
    csv_dataset_path = pickle_to_formatted_csv(dataset_path)

    RF = autoclass('webreduce.extension.classification.TableClassificationUtils')
    rf_classifier = RF(random_forest_model_path)
    rf_classifier.benchmarkTableClassification(csv_dataset_path)

def classify_table_2_phase(table_html):
    TableClassifier = autoclass('webreduce.extension.classification.TableClassifier')
    table_classifier = TableClassifier()
    return table_classifier.classify_table(table_html)

if __name__ == '__main__':
    with open(f'{os.path.dirname(os.path.realpath(__file__))}/../resources/table.html') as f:
        table_html = f.read()
        print(classify_table_2_phase(table_html))
