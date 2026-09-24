import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import os
import csv

# define every case
data_types = ['Overall', 'N400', 'P600']
faculty_ind = {'entity':2,'dependency':3,'pos':4,'tfidf':5}
subNo_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
artNo_list = [1, 2, 3, 4, 5, 6, 7, 8]

def com_RSA_by_n_timepoints(timepoints, data_type, model_name, faculty):
    # load model encoder hidden states
    encoder_hidden_states = np.load(f'../hidden_states/{model_name}_encoder_align.npy')
    # print(encoder_hidden_states.shape)

    # select encoder hidden states by factuality type
    faculty_encoder = []
    nofaculty_encoder = []
    faculty_data = []
    with open('./faculty_data.csv', 'r') as file:
        reader = csv.reader(file)
        ind = faculty_ind[faculty]
        for i in reader:
            faculty_data.append(i[ind])
    faculty_data = np.array(faculty_data).astype(int)
    for i in encoder_hidden_states:
        fal_temp = []
        no_temp = []
        for j in range(len(i)):
            if faculty_data[j]:
                fal_temp.append(i[j])
            else:
                no_temp.append(i[j])
        faculty_encoder.append(fal_temp)
        nofaculty_encoder.append(no_temp)
    faculty_encoder = np.array(faculty_encoder)
    nofaculty_encoder = np.array(nofaculty_encoder)
    # print(faculty_encoder.shape)
    # print(nofaculty_encoder.shape)

    # initialize RSM for diffent word type by its shape
    faculty_hidden_similarity_matrixes = np.zeros((faculty_encoder.shape[0], faculty_encoder.shape[1], faculty_encoder.shape[1]))
    nofaculty_hidden_similarity_matrixes = np.zeros((nofaculty_encoder.shape[0], nofaculty_encoder.shape[1], nofaculty_encoder.shape[1]))


    # compute factuality RSM by cosine similarity
    for i in range(faculty_encoder.shape[0]):
        current_data = faculty_encoder[i]

        dot_product = np.dot(current_data, current_data.T)
        norm = np.linalg.norm(current_data, axis=1, keepdims=True)
        similarity_matrix = dot_product / (norm * norm.T)

        faculty_hidden_similarity_matrixes[i] = similarity_matrix

    print(f"get_faculty_hiddens_similarity_matrixes output shape: {faculty_hidden_similarity_matrixes.shape}")

    # compute nonfactuality RSM by cosine similarity
    for i in range(nofaculty_encoder.shape[0]):
        current_data = nofaculty_encoder[i]

        dot_product = np.dot(current_data, current_data.T)
        norm = np.linalg.norm(current_data, axis=1, keepdims=True)
        similarity_matrix = dot_product / (norm * norm.T)

        nofaculty_hidden_similarity_matrixes[i] = similarity_matrix

    print(f"get_nofaculty_hiddens_similarity_matrixes output shape: {nofaculty_hidden_similarity_matrixes.shape}")

    # compute EEG RSM and Spearman corr for every subjects
    for subj_num in range(1,14):
        EEG_data_all = []
        # load EEG data from every article and concatenate them
        for art in artNo_list:
            EEG_data = np.load(f'../npy_bart2/art_{art}/{data_type}/sub_{subj_num}_art_{art}_epo.npy')
            EEG_data_all.append(EEG_data)

        EEG_data_all = np.concatenate(EEG_data_all, axis=0)

        # select EEG data by factuality type
        faculty_EEG = []
        nofaculty_EEG = []
        for i in range(len(EEG_data_all)):
            if faculty_data[i]:
                faculty_EEG.append(EEG_data_all[i])
            else:
                nofaculty_EEG.append(EEG_data_all[i])
        faculty_EEG = np.array(faculty_EEG)
        nofaculty_EEG = np.array(nofaculty_EEG)
        print(faculty_EEG.shape)
        print(nofaculty_EEG.shape)

        # reshape two EEG data by segmentation and computation requirement
        faculty_time_len = faculty_EEG.shape[-1]
        faculty_del_len = faculty_time_len % timepoints
        # print(type(time_len), type(del_len), type(timepoints))
        if faculty_del_len != 0:
            faculty_EEG_reshape = faculty_EEG[:,:,:-faculty_del_len].reshape(faculty_EEG.shape[0], faculty_EEG.shape[1], int(faculty_time_len/timepoints), timepoints)
        else:
            faculty_EEG_reshape = faculty_EEG.reshape(faculty_EEG.shape[0], faculty_EEG.shape[1], int(faculty_time_len/timepoints), timepoints)
        faculty_EEG_reshape = np.transpose(faculty_EEG_reshape, (2, 0, 3, 1))
        faculty_EEG_reshape = faculty_EEG_reshape.reshape(faculty_EEG_reshape.shape[0], faculty_EEG_reshape.shape[1], faculty_EEG_reshape.shape[2]*faculty_EEG_reshape.shape[3])
        print(faculty_EEG_reshape.shape)
        nofaculty_time_len = nofaculty_EEG.shape[-1]
        nofaculty_del_len = nofaculty_time_len % timepoints
        # print(type(time_len), type(del_len), type(timepoints))
        if nofaculty_del_len != 0:
            nofaculty_EEG_reshape = nofaculty_EEG[:,:,:-nofaculty_del_len].reshape(nofaculty_EEG.shape[0], nofaculty_EEG.shape[1], int(nofaculty_time_len/timepoints), timepoints)
        else:
            nofaculty_EEG_reshape = nofaculty_EEG.reshape(nofaculty_EEG.shape[0], nofaculty_EEG.shape[1], int(nofaculty_time_len/timepoints), timepoints)
        nofaculty_EEG_reshape = np.transpose(nofaculty_EEG_reshape, (2, 0, 3, 1))
        nofaculty_EEG_reshape = nofaculty_EEG_reshape.reshape(nofaculty_EEG_reshape.shape[0], nofaculty_EEG_reshape.shape[1], nofaculty_EEG_reshape.shape[2]*nofaculty_EEG_reshape.shape[3])
        print(nofaculty_EEG_reshape.shape)

        # compute factuality EEG RSM by pearman
        faculty_EEG_similarities_matrixes = []
        faculty_EEG_similarity = []
        for i in faculty_EEG_reshape:
            faculty_EEG_similarity_matrix = np.corrcoef(i, rowvar=True)
            faculty_EEG_similarity.append(faculty_EEG_similarity_matrix)
        faculty_EEG_similarities_matrixes.append(faculty_EEG_similarity)
        faculty_EEG_similarities_matrixes = np.array(faculty_EEG_similarities_matrixes)
        # np.save(f'EEG_similarities_matrices_Overall.npy', EEG_similarities_matrices)
        print(f"get_faculty_EEG_similarity_matrices output shape: {faculty_EEG_similarities_matrixes.shape}")

        # compute nonfactuality EEG RSM by pearman
        nofaculty_EEG_similarities_matrixes = []
        nofaculty_EEG_similarity = []
        for i in nofaculty_EEG_reshape:
            nofaculty_EEG_similarity_matrix = np.corrcoef(i, rowvar=True)
            nofaculty_EEG_similarity.append(nofaculty_EEG_similarity_matrix)
        nofaculty_EEG_similarities_matrixes.append(nofaculty_EEG_similarity)

        nofaculty_EEG_similarities_matrixes = np.array(nofaculty_EEG_similarities_matrixes)
        # np.save(f'EEG_similarities_matrices_Overall.npy', EEG_similarities_matrices)
        print(f"get_nofaculty_EEG_similarity_matrices output shape: {nofaculty_EEG_similarities_matrixes.shape}")

        # compute spearman corr between factuality encoder RSM and factuality EEG RSM
        faculty_spearman_corr_all = []
        faculty_p_value_all = []
        faculty_EEG_similarities_matrice = faculty_EEG_similarities_matrixes[0]
        for i in range(faculty_hidden_similarity_matrixes.shape[0]):
            spearman_corr_subj = []
            p_value_subj = []

            for j in range(faculty_EEG_similarities_matrice.shape[0]):
                hidden_similarity = faculty_hidden_similarity_matrixes[i]
                EEG_similarity = faculty_EEG_similarities_matrice[j]

                hidden_similarity_upper = np.triu(hidden_similarity, k=1)
                hidden_similarity_flatten = hidden_similarity_upper[hidden_similarity_upper != 0]

                EEG_similarity_upper = np.triu(EEG_similarity, k=1)
                EEG_similarity_flatten = EEG_similarity_upper[EEG_similarity_upper != 0]
                # print(hidden_similarity_flatten.shape, EEG_similarity_flatten.shape)
                spearman_corr, p_value = spearmanr(hidden_similarity_flatten, EEG_similarity_flatten)

                spearman_corr_subj.append(spearman_corr)
                p_value_subj.append(p_value)

            faculty_spearman_corr_all.append(spearman_corr_subj)
            faculty_p_value_all.append(p_value_subj)

        faculty_spearman_corr_all = np.array(faculty_spearman_corr_all)
        faculty_p_value_all = np.array(faculty_p_value_all)

        print(f"get_spearman output shapes:")
        print(f"faculty_spearman_corr_all: {faculty_spearman_corr_all.shape}")
        print(f"faculty_p_value_all: {faculty_p_value_all.shape}")

        np.save(f'../spearman_corr/spearman/{model_name}_{faculty}_{timepoints}time_course_spearman_corr_{data_type}_{subj_num}.npy', faculty_spearman_corr_all)
        np.save(f'../spearman_corr/p_value/{model_name}_{faculty}_{timepoints}time_course_p_value_{data_type}_{subj_num}.npy', faculty_p_value_all)

        # compute spearman corr between nonfactuality encoder RSM and nonfactuality EEG RSM
        nofaculty_spearman_corr_all = []
        nofaculty_p_value_all = []
        nofaculty_EEG_similarities_matrice = nofaculty_EEG_similarities_matrixes[0]

        for i in range(nofaculty_hidden_similarity_matrixes.shape[0]):
            spearman_corr_subj = []
            p_value_subj = []

            for j in range(nofaculty_EEG_similarities_matrice.shape[0]):
                hidden_similarity = nofaculty_hidden_similarity_matrixes[i]
                EEG_similarity = nofaculty_EEG_similarities_matrice[j]

                hidden_similarity_upper = np.triu(hidden_similarity, k=1)
                hidden_similarity_flatten = hidden_similarity_upper[hidden_similarity_upper != 0]

                EEG_similarity_upper = np.triu(EEG_similarity, k=1)
                EEG_similarity_flatten = EEG_similarity_upper[EEG_similarity_upper != 0]
                # print(hidden_similarity_flatten.shape, EEG_similarity_flatten.shape)
                spearman_corr, p_value = spearmanr(hidden_similarity_flatten, EEG_similarity_flatten)

                spearman_corr_subj.append(spearman_corr)
                p_value_subj.append(p_value)

            nofaculty_spearman_corr_all.append(spearman_corr_subj)
            nofaculty_p_value_all.append(p_value_subj)

        nofaculty_spearman_corr_all = np.array(nofaculty_spearman_corr_all)
        nofaculty_p_value_all = np.array(nofaculty_p_value_all)

        print(f"get_spearman output shapes:")
        print(f"faculty_spearman_corr_all: {nofaculty_spearman_corr_all.shape}")
        print(f"faculty_p_value_all: {nofaculty_p_value_all.shape}")

        np.save(f'../spearman_corr/spearman/{model_name}_no{faculty}_{timepoints}time_course_spearman_corr_{data_type}_{subj_num}.npy', nofaculty_spearman_corr_all)
        np.save(f'../spearman_corr/p_value/{model_name}_no{faculty}_{timepoints}time_course_p_value_{data_type}_{subj_num}.npy', nofaculty_p_value_all)



def faculty_analysis(model_name, timepoints, faculty):
    for data_type in data_types:
        com_RSA_by_n_timepoints(timepoints, data_type, model_name, faculty)