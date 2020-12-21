# Web table classification


## Goldstandard_Generation

Contains scripts for data collection and image rendering to generate a new gold standard

## Data

Goldstandard datasets can be downloaded here: http://data.dws.informatik.uni-mannheim.de/visual_table_classification/

- GoldStandard_tvt: Goldstandard images in training, validation and test split sub folders
- gs_125_warc_files_comb.pkl: goldstandard html code, url, s3links, labels
- predictions2-with-features.pkl: manual features and predictions by DWTC-extractor classifier for new gold standard
- Helper files: all_test_ids.npy, all_val_ids.npy

## Model_development

Contains scripts and saved models of tested classifiers

### Baseline

Contains scripts for Evaluation of DWTC classifier on new data and retrained Random Forest classifier

### CNN_classifier

Contains image classification with VGG16 and ResNet15 architectures

### Feature_extractor

Contains visual feature extraction, individual feature datasets and Random Forest classification with visual and manual features


## Runtime_testing

Contains scripts to test runtime of best performing classifiers (runtime test for DWTC extractor classifier can be found in dwtc-extension folder)

- DWTC extractor Random Forest classification
- Baseline Random Forest classification pipeline with manual features from DWTC extractor
- ResNet finetuned CNN classification pipeline
- Random Forest classification pipeline with VGG16 lower level features and manual features from DWTC extractor

### Run tests

1. To create executable jar with dwtc-extension java classifier and copy it to runtime testing resources run:
```
./create_jar.sh
```

2. To execute all runtime tests with different batch sizes run:
```
python3.8 runtime_testing/run.py
```
Alternatively, run designated python scripts to execute runtime tests for individual classifier only. The execution times will be written to stdout and for all Python classifiers also to timing.csv.

## Visualization_notebooks

Contains Jupyter Notebooks used for visualization of VGG16 and ResNet50 feature maps, and missclassification comparison

## dwtc-extension

Contains scripts for accessing DWTC extractor 
DWTC extractor github repository: https://github.com/JulianEberius/dwtc-extractor
