import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import os

# define every case
data_types = ['Overall', 'N400', 'P600']
subNo_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
artNo_list = [1, 2, 3, 4, 5, 6, 7, 8]

# compute RSA after segmenting by n timepoints in which n can be any timepoints
def com_RSA_by_n_timepoints(timepoints, data_type, model_name):
    # load model encoder hidden states
    encoder_hidden_states = np.load(f'../hidden_states/{model_name}_encoder_align.npy')
    # initialize RSM of hidden states
    if model_name == 't5_base' or model_name == 'bart' or model_name == 'pegasus_base':
        hiddens_similarity_matirixes = np.zeros((12, 1459, 1459))
    elif model_name == 'pegasus':
        hiddens_similarity_matirixes = np.zeros((16, 1459, 1459))
    elif model_name == 't5_small' or model_name == 'bart_base':
        hiddens_similarity_matirixes = np.zeros((6, 1459, 1459))
    else:
        hiddens_similarity_matirixes = np.zeros((24, 1459, 1459))

    # compute encoder RSM by cosine similarity
    for i in range(encoder_hidden_states.shape[0]):
        current_data = encoder_hidden_states[i]

        dot_product = np.dot(current_data, current_data.T)
        norm = np.linalg.norm(current_data, axis=1, keepdims=True)
        similarity_matrix = dot_product / (norm * norm.T)

        hiddens_similarity_matirixes[i] = similarity_matrix

    print(f"get_hiddens_similarity_matirxes output shape: {hiddens_similarity_matirixes.shape}")

    # compute EEG RSM and Spearman corr for every subjects
    for subj_num in range(1,14):
        EEG_data_all = []
        # load EEG data from every article and concatenate them
        for art in artNo_list:
            EEG_data = np.load(f'../npy_bart2/art_{art}/{data_type}/sub_{subj_num}_art_{art}_epo.npy')
            EEG_data_all.append(EEG_data)

        EEG_data_all = np.concatenate(EEG_data_all, axis=0)

        # reshape EEG data by segmentation and computation requirement
        time_len = EEG_data_all.shape[-1]
        del_len = time_len % timepoints
        # print(type(time_len), type(del_len), type(timepoints))
        print(EEG_data_all.shape)
        if del_len != 0:
            EEG_data_reshape = EEG_data_all[:,:,:-del_len].reshape(EEG_data_all.shape[0], EEG_data_all.shape[1], int(time_len/timepoints), timepoints)
        else:
            EEG_data_reshape = EEG_data_all.reshape(EEG_data_all.shape[0], EEG_data_all.shape[1], int(time_len/timepoints), timepoints)
        EEG_data_reshape = np.transpose(EEG_data_reshape, (2, 0, 3, 1))
        EEG_data_reshape = EEG_data_reshape.reshape(EEG_data_reshape.shape[0], EEG_data_reshape.shape[1], EEG_data_reshape.shape[2]*EEG_data_reshape.shape[3])
        print(EEG_data_reshape.shape)

        # compute EEG RSM by pearman
        EEG_similarities_matrices = []

        EEG_similarity = []
        for i in EEG_data_reshape:
            EEG_similarity_matrix = np.corrcoef(i, rowvar=True)
            EEG_similarity.append(EEG_similarity_matrix)
        EEG_similarities_matrices.append(EEG_similarity)

        EEG_similarities_matrices = np.array(EEG_similarities_matrices)
        # np.save(f'EEG_similarities_matrices_Overall.npy', EEG_similarities_matrices)
        print(f"get_EEG_similarity_matrices output shape: {EEG_similarities_matrices.shape}")

        # compute spearman corr between encoder RSM and EEG RSM
        spearman_corr_all = []
        p_value_all = []
        EEG_similarities_matrice = EEG_similarities_matrices[0]

        for i in range(hiddens_similarity_matirixes.shape[0]):
            spearman_corr_subj = []
            p_value_subj = []

            for j in range(EEG_similarities_matrice.shape[0]):
                hidden_similarity = hiddens_similarity_matirixes[i]
                EEG_similarity = EEG_similarities_matrice[j]

                hidden_similarity_upper = np.triu(hidden_similarity, k=1)
                hidden_similarity_flatten = hidden_similarity_upper[hidden_similarity_upper != 0]

                EEG_similarity_upper = np.triu(EEG_similarity, k=1)
                EEG_similarity_flatten = EEG_similarity_upper[EEG_similarity_upper != 0]
                # print(hidden_similarity_flatten.shape, EEG_similarity_flatten.shape)
                spearman_corr, p_value = spearmanr(hidden_similarity_flatten, EEG_similarity_flatten)

                spearman_corr_subj.append(spearman_corr)
                p_value_subj.append(p_value)

            spearman_corr_all.append(spearman_corr_subj)
            p_value_all.append(p_value_subj)

        spearman_corr_all = np.array(spearman_corr_all)
        p_value_all = np.array(p_value_all)

        print(f"get_spearman output shapes:")
        print(f"spearman_corr_all: {spearman_corr_all.shape}")
        print(f"p_value_all: {p_value_all.shape}")

        np.save(f'../spearman_corr/spearman/{model_name}_{timepoints}time_course_spearman_corr_{data_type}_{subj_num}.npy', spearman_corr_all)
        np.save(f'../spearman_corr/p_value/{model_name}_{timepoints}time_course_p_value_{data_type}_{subj_num}.npy', p_value_all)

def n_timepoints_analysis(model_name, timepoints):
    for data_type in data_types:
        com_RSA_by_n_timepoints(timepoints, data_type, model_name)