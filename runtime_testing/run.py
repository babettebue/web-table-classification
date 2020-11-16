import os
import pandas as pd
import Re_trained_RF_pipeline
import ResNetCNN_pipeline
import VGG16_feature_extractor_pipeline
from utils import dwtc_wrapper
from utils.image_rendering import html_to_images, pickle_to_formatted_csv

resources = os.path.join('runtime_testing', 'resources')
dataset_path = os.path.join(resources, 'performance_testing_1500_tables.pkl')
ResNet_weights_path = os.path.join(resources, 'resnet_model1_finetuned.h5')
VGG16_model_path = os.path.join(resources, 'heuristic_vgg16_feature_maps_rf.joblib')
retrained_model_path = os.path.join(resources, 'heuristic_rf.joblib')
random_forest_model_path = os.path.join(resources, 'RandomForest_P1.mdl')

BENCHMARK_TEST_BATCH_SIZES = [10, 100, 1000]
BENCHMARK_ITERATIONS = 3

def run_benchmarking_suite():
    df = pd.read_pickle(dataset_path)
    renderable_df = filter_renderable_tables(df)
    # renderable_df = pd.read_pickle(os.path.join(resources, 'renderable_tables.pkl'))
    sample_dataset_path = os.path.join(resources, 'sample.pkl')

    for iteration_size in BENCHMARK_TEST_BATCH_SIZES:
        print(f'\n\n||||| ############ Running Benchmarking Tests for {iteration_size} HTML Tables ############')
        sample = renderable_df.sample(iteration_size)
        sample.to_pickle(sample_dataset_path)
        # sample = pd.read_pickle(os.path.join(resources, 'sample.pkl'))
        for _ in range(BENCHMARK_ITERATIONS):
            ResNetCNN_pipeline.make_prediction(sample, ResNet_weights_path)
            VGG16_feature_extractor_pipeline.make_prediction(sample, sample_dataset_path, VGG16_model_path, random_forest_model_path)
            dwtc_wrapper.make_prediction(sample_dataset_path, random_forest_model_path)
            Re_trained_RF_pipeline.make_prediction(sample_dataset_path, retrained_model_path, random_forest_model_path)


def filter_renderable_tables(df):
    # filter out html tables which cannot be rendered completely (e.g. due to changes in the website)
    ids, _ = html_to_images(df)
    renderable_df = df[df['id'].isin(ids)]
    renderable_df.to_pickle(os.path.join(resources, 'renderable_tables.pkl'))    
    print(f"{len(renderable_df.index)} tables could successfully be rendered.")
    
    return renderable_df


if __name__ == '__main__':
    run_benchmarking_suite()
