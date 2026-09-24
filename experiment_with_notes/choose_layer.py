import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# define every case
model_names = ['pegasus_base', 'pegasus', 'bart_base', 'bart', 't5_small', 't5_base', 't5_large', 't5_3b', 't5_11b']
channels = ['Overall'] # ['Overall', 'N400', 'P600']
timepoints = [1] # [1, 5, 25]


def choose_layer():
    result = {}

    # load significance result for every case
    for model_name in model_names:
        for channel in channels:
            for timepoint in timepoints:
                spearman_data = []
                choose_jud = np.load(f'../significance/{model_name}_{timepoint}_{channel}.npy')
                # load spearman data from every subjects
                for i in range(1,14):
                    sp_data = np.load(f'../spearman_corr/spearman/{model_name}_{timepoint}time_course_spearman_corr_{channel}_{i}.npy')
                    spearman_data.append(sp_data)
                spearman_data = np.transpose(np.array(spearman_data), (1, 0, 2))
                spearman_sum = []
                # sum spearman data by significance result
                for i in range(spearman_data.shape[0]):
                    temp = 0.0
                    count = 0
                    for j in range(spearman_data.shape[2]):
                        if choose_jud[i][j]:
                            temp += np.mean(spearman_data[i,:,j])
                    # spearman_sum.append([temp,count])
                    spearman_sum.append(temp)
                # select the layer which significant spearman sum is the biggest
                spearman_sum = np.array(spearman_sum)
                max_data = np.max(spearman_sum)
                ind = np.where(spearman_sum == max_data)
                # result[f'{model_name}_{timepoint}_{channel}'] = spearman_sum
                result[f'{model_name}_{timepoint}_{channel}'] = int(ind[0][0])
    print(result)

    # save result in json file
    with open('./model_choose.json', 'w') as json_file:
        json.dump(result, json_file)



def choose_plot_draw():
    # load choosed layer data
    with open('./model_choose.json', 'r') as file:
        choose_result = json.load(file)

    for timepoint in timepoints:
        overall_draw_data = []
        N400_draw_data = []
        P600_draw_data = []
        # load spearman data for every case
        for model_name in model_names:
            for channel in channels:
                all_data = []
                for i in range(1,14):
                    sp_data = np.load(f'../spearman_corr/spearman/{model_name}_{timepoint}time_course_spearman_corr_{channel}_{i}.npy')
                    all_data.append(sp_data)
                all_data = np.mean(np.array(all_data), axis=0)
                ind = int(choose_result[f'{model_name}_{timepoint}_{channel}'])
                if channel == 'Overall':
                    overall_draw_data.append(all_data[ind])
                elif channel == 'N400':
                    N400_draw_data.append(all_data[ind])
                elif channel == 'P600':
                    P600_draw_data.append(all_data[ind])
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']
        plt.figure(figsize=(10,6))
        # draw choosed spearman plot
        for i in range(8):
            plt.plot(overall_draw_data[i], color=colors[i], label=model_names[i])

        plt.title(f'RSA Score in {timepoint}point')
        plt.xlabel('Time')
        plt.ylabel('RSA Score')

        plt.legend()
        plt.savefig(f'../plots/{timepoint}points_plot.png', dpi=300)
        plt.show()
