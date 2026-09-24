import numpy as np
from scipy import stats
import os
from glob import glob
import matplotlib.pyplot as plt

# define every case
channels = ['Overall']  # ['Overall', 'N400', 'P600']
model_names = ['pegasus_base', 'pegasus', 'bart_base', 'bart', 't5_small', 't5_base', 't5_large', 't5_3b', 't5_11b']
timepoints = [1] # [1, 5, 25]
factualitys = ['entity', 'noentity', 'dependency', 'nodependency', 'pos', 'nopos', 'tfidf', 'notfidf']

def calculate_significance(folder_path, file_pattern, output_file):
    # get every file name by target condition
    file_path_pattern = os.path.join(folder_path, file_pattern)
    files = sorted(glob(file_path_pattern))

    if len(files) == 0:
        print(f"erro:can't find {file_pattern} in {folder_path}.")
        return

    print(f"find {len(files)} files.")

    # read first file to get shape
    first_data = np.load(files[0])
    layers, time_points = first_data.shape

    # initialize result matrix
    result = np.zeros((layers, time_points), dtype=int)

    # compute in every timpoint and every layer
    for layer in range(layers):
        for time in range(time_points):
            # get data for every subjects in this layer and this timepoint
            data_point = []
            for file in files:
                data = np.load(file)
                data_point.append(data[layer, time])

            # Wilcoxon Signed-Rank Test
            statistic, p_value = stats.wilcoxon(data_point)

            # if p lower than 0.05 it's significant and set it 1
            if p_value < 0.05:
                result[layer, time] = 1

    # save result
    np.save(output_file, result)
    print(f"result save to {output_file}")

    return result

import numpy as np

def plot_significance(result, model_name, channel, timepoint, factuality):
    layers, time_points = result.shape

    plt.figure(figsize=(15, 10))

    for layer in range(layers):
        # data of siginficant point times its layer num
        layer_data = result[layer] * (layer + 1)
        plt.scatter(range(time_points), layer_data, s=1, label=f'Layer {layer + 1}')

    plt.xlabel('Time Points (ms)')
    plt.ylabel('Layer * Significance')
    plt.title(f'{model_name}_{factuality}_{timepoint}_{channel} Significance across Layers and Time Points')
    # plt.legend()
    plt.grid(True)
    plt.ylim(0, layers + 1)

    x_ticks = range(-200, 1501, 100)
    plt.xticks([(x + 200) // 4 for x in x_ticks], x_ticks)
    plt.xlim(-12.5, 387.5)  # (-250 + 200) / 4 到 (1550 + 200) / 4

    # save
    plt.savefig(f'../factuality_significance/{model_name}_{factuality}_{timepoint}_{channel}.png', dpi=300)
    print(f"plot saved in ../factuality_significance/{model_name}_{factuality}_{timepoint}_{channel}.png")

    plt.show()

def significance(model_name, channel, timepoint, factuality):
    folder_path = '../spearman_corr/spearman'
    file_pattern = f'{model_name}_{factuality}_{timepoint}time_course_spearman_corr_{channel}_*.npy'
    output_file = f'../factuality_significance/{model_name}_{factuality}_{timepoint}_{channel}.npy'

    # Process the file and save the result
    result = calculate_significance(folder_path, file_pattern, output_file)

    # Draw plots
    plot_significance(result[::-1], model_name, channel, timepoint, factuality)

def all_run():
    for model_name in model_names:
        for channel in channels:
            for timepoint in timepoints:
                for factuality in factualitys:
                    significance(model_name, channel, timepoint, factuality)