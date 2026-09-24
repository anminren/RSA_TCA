import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

model_names = ['pegasus_base', 'pegasus', 'bart_base', 'bart', 't5_small', 't5_base', 't5_large', 't5_3b', 't5_11b']
channels = ['Overall'] #['Overall', 'N400', 'P600']
timepoints = [1] # [1, 5, 25]
factualitys = ['dependency', 'entity', 'pos', 'tfidf']

def compute_SigArea(model_name, channel, timepoint):
    # get significance for every timepoint in every layer
    significance_result = np.load(f'../significance/{model_name}_{timepoint}_{channel}.npy')
    spearman_data = []
    # load spearman corr of 13 subjects and get their sum
    for i in range(1,14):
        spearman = np.load(f'../spearman_corr/spearman/{model_name}_{timepoint}time_course_spearman_corr_{channel}_{i}.npy')
        if spearman_data == []:
            spearman_data = spearman
        else:
            spearman_data += spearman
    area_result = []
    # compute sig area by significance
    for i in range(len(spearman_data)):
        area = 0.0
        significance = significance_result[i]
        for j in range(len(significance)-1):
            # if its significance equal to 1 and its latter equal to 1 add this area
            if significance[j] == 1 and significance[j+1] == 1:
                area += np.trapz(spearman_data[i][j:j+2], [0,1])
        area_result.append(area)
    # print(area_result)
    return area_result

def factuality_compute_SigArea(model_name, channel, timepoint, factuality):
    # get significance for every timepoint in every layer with different factuality type
    factuality_significance_result = np.load(f'../factuality_significance/{model_name}_{factuality}_{timepoint}_{channel}.npy')
    nofactuality_significance_result = np.load(f'../factuality_significance/{model_name}_no{factuality}_{timepoint}_{channel}.npy')
    factuality_spearman_data = []
    nofactuality_spearman_data = []
    # load spearman corr of 13 subjects and get their sum with different factuality type
    for i in range(1,14):
        factuality_spearman = np.load(f'../spearman_corr/spearman/{model_name}_{factuality}_{timepoint}time_course_spearman_corr_{channel}_{i}.npy')
        nofactuality_spearman = np.load(f'../spearman_corr/spearman/{model_name}_no{factuality}_{timepoint}time_course_spearman_corr_{channel}_{i}.npy')
        if factuality_spearman_data == []:
            factuality_spearman_data = factuality_spearman
            nofactuality_spearman_data = nofactuality_spearman
        else:
            factuality_spearman_data += factuality_spearman
            nofactuality_spearman_data += nofactuality_spearman
    factuality_area_result = []
    nofactuality_area_result = []
    # compute sig area by significance
    # for factuality type
    for i in range(len(factuality_spearman_data)):
        area = 0.0
        significance = factuality_significance_result[i]
        for j in range(len(significance)-1):
            if significance[j] == 1 and significance[j+1] == 1:
                area += np.trapz(factuality_spearman_data[i][j:j+2], [0,1])
        factuality_area_result.append(area)
    # for nonfactuality type
    for i in range(len(nofactuality_spearman_data)):
        area = 0.0
        significance = nofactuality_significance_result[i]
        for j in range(len(significance)-1):
            if significance[j] == 1 and significance[j+1] == 1:
                area += np.trapz(nofactuality_spearman_data[i][j:j+2], [0,1])
        nofactuality_area_result.append(area)
    # print(area_result)
    return factuality_area_result, nofactuality_area_result

# run compute_SigArea for every case
def all_run():
    result = {}
    for model_name in model_names:
        for channel in channels:
            for timepoint in timepoints:
                result[f"{model_name}_{channel}_{timepoint}"] = compute_SigArea(model_name, channel, timepoint)
    factuality_result = {}
    for model_name in model_names:
        for channel in channels:
            for timepoint in timepoints:
                for factuality in factualitys:
                    tmp = factuality_compute_SigArea(model_name, channel, timepoint, factuality)
                    factuality_result[f"{model_name}_{factuality}_{channel}_{timepoint}"] = tmp[0]
                    factuality_result[f"{model_name}_no{factuality}_{channel}_{timepoint}"] = tmp[1]
    with open('./sig_area.json', 'w') as json_file:
        json.dump({'all_word':result, 'factuality_word': factuality_result}, json_file)
