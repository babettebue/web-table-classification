import os
import logging
import urllib.request


jar_file_path = f'{os.path.dirname(os.path.realpath(__file__))}/../runtime_testing/resources/dwtc-extension-1.0-jar-with-dependencies.jar'
if not os.path.isfile(jar_file_path):
    logging.warning('JAR File missing, downloading')
    urllib.request.urlretrieve(
        "https://github.com/lavuy/web-table-classification/raw/c6cc1eeb62b996b8bbcd26b6dc27841d1464b884/runtime_testing/resources/dwtc-extension-1.0-jar-with-dependencies.jar",
        jar_file_path)

os.environ['CLASSPATH'] = jar_file_path

global table_classifier
table_classifier = None


def _init_table_classifier():
    from jnius import autoclass

    TableClassifier = autoclass('webreduce.extension.classification.TableClassifier')
    global table_classifier
    table_classifier = TableClassifier()


def classify_table_2_phase(table_html):
    if table_classifier is None:
        _init_table_classifier()
    return table_classifier.classifyTable(table_html)


def classify_and_print(table_html):
    print(f'{os.getpid()}: {classify_table_2_phase(table_html).tableType.toString()}')
    return classify_table_2_phase(table_html).tableType.toString()


if __name__ == '__main__':
    with open(f'{os.path.dirname(os.path.realpath(__file__))}/../runtime_testing/resources/table.html') as f:
        table_html = f.read()
        print(classify_table_2_phase(table_html).tableType.toString())

# import multiprocessing
# from concurrent.futures import ProcessPoolExecutor
# from datetime import datetime
# if __name__ == '__main__':
#     multiprocessing.set_start_method("spawn", force=True)
#     start = datetime.now()
#     futures = []
#     with ProcessPoolExecutor(max_workers=1
#                              ) as pool:
#         with open(f'{os.path.dirname(os.path.realpath(__file__))}/../runtime_testing/resources/table.html') as f:
#             table_html = f.read()
#             for _ in range(1000):
#                 futures.append(pool.submit(classify_and_print, table_html))
#                 # futures.append(pool.submit(classify_table_2_phase, table_html))
#                 # print(classify_table_2_phase(table_html).tableType.toString())
#     for future in futures:
#         res = future.result()
#         print(res)
#         assert res
#         # finished += 1
#         # R logging.info(f"Finished {finished}/{len(futures)} overall")
#     print(f"Took: {datetime.now() - start}")
