import os
import logging
import urllib.request

jar_file_path = f'{os.path.dirname(os.path.realpath(__file__))}/../runtime_testing/resources/dwtc-extension-1.0-jar-with-dependencies.jar'
if not os.path.isfile(jar_file_path):
    logging.warning('JAR File missing, downloading')


    urllib.request.urlretrieve("https://github.com/lavuy/web-table-classification/raw/c6cc1eeb62b996b8bbcd26b6dc27841d1464b884/runtime_testing/resources/dwtc-extension-1.0-jar-with-dependencies.jar", jar_file_path)

os.environ['CLASSPATH'] = jar_file_path

#import jnius_config

# if not jnius_config.vm_running:
#     jnius_config.set_classpath(jar_file_path)
from jnius import autoclass

TableClassifier = autoclass('webreduce.extension.classification.TableClassifier')
table_classifier = TableClassifier()


def classify_table_2_phase(table_html):
    return table_classifier.classifyTable(table_html)


if __name__ == '__main__':
    with open(f'{os.path.dirname(os.path.realpath(__file__))}/../runtime_testing/resources/table.html') as f:
        table_html = f.read()
        print(classify_table_2_phase(table_html).tableType.toString())
