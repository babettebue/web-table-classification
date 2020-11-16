# Web table classification

##dwtc-extension

Contains scripts to access DWTC extractor 
DWTC extractor github repository: https://github.com/JulianEberius/dwtc-extractor


## goldstandard_generation

Contains scripts for data collection and image rendering to generate a new gold standard


## GoldStandard_tvt

Contains Goldstandard images in training, validation and test split sub folders

## data

Contains Goldstandard pkl and csv datasets 

- gs_125_warc_files_comb.pkl: goldstandard html code, url, s3links, labels
- predictions2-with-features.pkl: manual features and predictions by DWTC-extractor classifier for new gold standard
- Helper files: all_test_ids.npy, all_val_ids.npy, 

## model_development

Contains scripts and saved models of tested classifiers

### Baseline

Contains scripts for Evaluation of DWTC classifier on new data and retrained Random Forest classifier

### CNN_classifier

Contains image classification with VGG16 and ResNet15 architectures

### Feature_extractor

Contains visual feature extraction, individual feature datasets and Random Forest classification with visual and manual features

## runtime_testing

Contains scripts to test runtime of best performing classifiers (runtime test for DWTC extractor classifier can be found in DWTC extractor folder)

- ResNet finetuned CNN classification pipeline
- Random Forest classification pipeline with VGG16 lower level features and manual features from DWTC extractor

## visualization_notebooks

Contains Jupyter Notebooks usef for visualization of VGG16 and ResNet50 feature maps, and missclassification comparison

