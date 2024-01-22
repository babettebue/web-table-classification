import jnius_config
import os
jar_file_path = f'{os.path.dirname(os.path.realpath(__file__))}/../resources/dwtc-extension-1.0-jar-with-dependencies.jar'
jnius_config.set_classpath(jar_file_path)

from jnius import autoclass
TableClassifier = autoclass('webreduce.extension.classification.TableClassifier')
table_classifier = TableClassifier()


def classify_table_2_phase(table_html):
    return table_classifier.classifyTable(table_html)

if __name__ == '__main__':
    with open(f'{os.path.dirname(os.path.realpath(__file__))}/../resources/table.html') as f:
        table_html = f.read()
        print(classify_table_2_phase(table_html).tableType.toString())
