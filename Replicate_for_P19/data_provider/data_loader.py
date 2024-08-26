import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import warnings
import torch
warnings.filterwarnings('ignore')

class P19Datasaets(Dataset):
    def __init__(self, idx_dataset, 
                 root_path = '/home/DAHS2/Timellm/Replicate_for_P19/',
                 PT_dict_path='dataset/data/processed_data/PT_dict_list_6.npy', 
                 outcomes_path = 'dataset/data/processed_data/arr_outcomes_6.npy'):

        # init
        self.split_id = idx_dataset #sample id set [Train, Valid, Test]
        self.seq_len = 60
        self.root_path = root_path
        self.outcomes_path = outcomes_path
        self.PT_dict_path = PT_dict_path
        self.tot_len = self.__read_data__()
        self.slide_unit = 1

        # self.tot_len = 100
        
    def __read_data__(self):
        
        Pdict_list = np.load(self.root_path + self.PT_dict_path, allow_pickle=True)
        arr_outcomes = np.load(self.root_path + self.outcomes_path, allow_pickle=True)
        
        Pdict_list = np.load(self.root_path + self.PT_dict_path, allow_pickle=True)
        arr_outcomes = np.load(self.root_path + self.outcomes_path, allow_pickle=True)
        
        self.Pdata = Pdict_list[self.split_id]
        self.Pdata_label = arr_outcomes[self.split_id]
        
        T, F = self.Pdata[0]['arr'].shape # 60, 34
        D = len(self.Pdata[0]['extended_static'])

        Ptrain_tensor = np.zeros((len(self.Pdata), T, F))
        Ptrain_static_tensor = np.zeros((len(self.Pdata), D))

        for i in range(len(self.Pdata)):
            Ptrain_tensor[i] = self.Pdata[i]['arr'] # num of patient, T, N
            Ptrain_static_tensor[i] = self.Pdata[i]['extended_static']

        mf, stdf = getStats(Ptrain_tensor)
        ms, ss = getStats_static(Ptrain_static_tensor, dataset='P19')

        self.Ptrain_tensor, self.Ptrain_static_tensor, self.Ptrain_time_tensor, self.ytrain_tensor = tensorize_normalize(self.Pdata, 
                                                                                                     self.Pdata_label,mf,
                                                                                                    stdf, ms, ss)
        
        self.tot_len = len(self.Ptrain_tensor)
        
        return self.tot_len

    def __getitem__(self, index):
        
        data_x = self.Ptrain_tensor[index][:, :int(self.Ptrain_tensor.shape[2] / 2)] # B, T, N
        data_y = self.ytrain_tensor[index]
        time = self.Ptrain_time_tensor[index]
        real_time = self.Pdata[index]['length'] 
        
        demo_list = []
        for d in range(len(self.Pdata[index]['extended_static'])):
            demo_list.append(self.Pdata[index]['extended_static'][d])
        demo = np.array(demo_list)
        
        return data_x, data_y, time, real_time, demo
        
    def __len__(self):
        return self.tot_len
    
    
class P19DatasaetsDownSampled(Dataset):
    def __init__(self, idx_dataset, 
                 root_path = '/home/DAHS2/Timellm/Replicate_for_P19/',
                 PT_dict_path='dataset/data/processed_data/PT_dict_list_6.npy', 
                 outcomes_path = 'dataset/data/processed_data/arr_outcomes_6.npy'):

        # init
        self.split_id = idx_dataset #sample id set [Train, Valid, Test]
        self.seq_len = 60
        self.root_path = root_path
        self.outcomes_path = outcomes_path
        self.PT_dict_path = PT_dict_path
        tot_len = self.__read_data__()
        self.slide_unit = 1
        
        self.tot_len = tot_len

    def __read_data__(self):
        
        Pdict_list = np.load(self.root_path + self.PT_dict_path, allow_pickle=True)
        arr_outcomes = np.load(self.root_path + self.outcomes_path, allow_pickle=True)
        
        self.Pdata = Pdict_list[self.split_id]
        self.Pdata_label = arr_outcomes[self.split_id]
        
        label_0_indices = np.where(self.Pdata_label == 0)[0]
        np.random.seed(42) 
        downsampled_indices = np.random.choice(label_0_indices, len(np.where(self.Pdata_label == 1)[0]), replace=False)
        
        Ptrain_downsampled = self.Pdata[downsampled_indices]
        Ptrain_label_downsampled = self.Pdata_label[downsampled_indices]

        label_1_indices = np.where(self.Pdata_label == 1)[0]
        # label_1_count = len(label_1_indices)
        # oversample_count = len(np.where(self.Pdata_label == 1)[0]) - label_1_count
        # oversampled_indices = np.random.choice(label_1_indices, oversample_count, replace=True)

        # Ptrain_oversampled = self.Pdata[oversampled_indices]
        # Ptrain_label_oversampled = self.Pdata_label[oversampled_indices]

        Ptrain_final = np.concatenate((Ptrain_downsampled, self.Pdata[label_1_indices]), axis=0)
        Ptrain_label_final = np.concatenate((Ptrain_label_downsampled, self.Pdata_label[label_1_indices]), axis=0)

        shuffle_indices = np.random.permutation(len(Ptrain_final))
        self.Ptrain_final = Ptrain_final[shuffle_indices]
        self.Ptrain_label_final = Ptrain_label_final[shuffle_indices]
        
        self.tot_len = len(self.Ptrain_final)
        
        return self.tot_len

    def __getitem__(self, index):
        demo_list = []
        real_measured = self.Ptrain_final[index]['length']
        real_data = self.Ptrain_final[index]['arr'][:real_measured]
        padded_data = self.Ptrain_final[index]['arr'][real_measured:]
        real_data[real_data == 0] = np.nan
        
        data_x = np.concatenate((real_data, padded_data), axis = 0)
        data_y = self.Ptrain_label_final[index]
        time = self.Ptrain_final[index]['time']
        for d in range(len(self.Ptrain_final[index]['extended_static'])):
            demo_list.append(self.Ptrain_final[index]['extended_static'][d])
        demo = np.array(demo_list)
        
        return data_x, data_y, time, real_measured, demo
        
    def __len__(self):
        return self.tot_len
    
class P19DatasaetsUpSampled(Dataset):
    def __init__(self, idx_dataset, 
                 root_path = '/home/DAHS2/Timellm/Replicate_for_P19/',
                 PT_dict_path='dataset/data/processed_data/PT_dict_list_6.npy', 
                 outcomes_path = 'dataset/data/processed_data/arr_outcomes_6.npy'):

        # init
        self.split_id = idx_dataset #sample id set [Train, Valid, Test]
        self.seq_len = 60
        self.root_path = root_path
        self.outcomes_path = outcomes_path
        self.PT_dict_path = PT_dict_path
        self.tot_len = self.__read_data__()
        self.slide_unit = 1

        # self.tot_len = 100

    def __read_data__(self):
        
        Pdict_list = np.load(self.root_path + self.PT_dict_path, allow_pickle=True)
        arr_outcomes = np.load(self.root_path + self.outcomes_path, allow_pickle=True)
        
        self.Pdata = Pdict_list[self.split_id]
        self.Pdata_label = arr_outcomes[self.split_id]
        
        label_0_indices = np.where(self.Pdata_label == 0)[0]
        label_1_indices = np.where(self.Pdata_label == 1)[0]

        # Oversampling of label 1
        label_1_count = len(label_1_indices)
        label_0_count = len(label_0_indices)
        oversample_count = label_0_count - label_1_count

        # Oversampling indices from label_1_indices with replacement
        oversampled_indices = np.random.choice(label_1_indices, oversample_count, replace=True)

        # 결합하여 최종 데이터셋 구성
        Ptrain_oversampled = self.Pdata[oversampled_indices]
        Ptrain_label_oversampled = self.Pdata_label[oversampled_indices]

        # 결합
        Ptrain_final = np.concatenate((Ptrain_oversampled, self.Pdata[label_0_indices], self.Pdata[label_1_indices]), axis=0)
        Ptrain_label_final = np.concatenate((Ptrain_label_oversampled, self.Pdata_label[label_0_indices], self.Pdata_label[label_1_indices]), axis=0)

        # 데이터 셔플링
        shuffle_indices = np.random.permutation(len(Ptrain_final))
        self.Ptrain_final = Ptrain_final[shuffle_indices]
        self.Ptrain_label_final = Ptrain_label_final[shuffle_indices]
        
        T, F = self.Ptrain_final[0]['arr'].shape # 60, 34
        D = len(self.Ptrain_final[0]['extended_static'])

        Ptrain_tensor = np.zeros((len(self.Ptrain_final), T, F))
        Ptrain_static_tensor = np.zeros((len(self.Ptrain_final), D))

        for i in range(len(self.Ptrain_final)):
            Ptrain_tensor[i] = self.Ptrain_final[i]['arr'] # num of patient, T, N
            Ptrain_static_tensor[i] = self.Ptrain_final[i]['extended_static']

        mf, stdf = getStats(Ptrain_tensor)
        ms, ss = getStats_static(Ptrain_static_tensor, dataset='P19')

        self.Ptrain_tensor, self.Ptrain_static_tensor, self.Ptrain_time_tensor, self.ytrain_tensor = tensorize_normalize(self.Ptrain_final, 
                                                                                                     self.Ptrain_label_final,mf,
                                                                                                    stdf, ms, ss)
        
        self.tot_len = len(self.Ptrain_tensor)
        
        return self.tot_len

    def __getitem__(self, index):
        
        data_x = self.Ptrain_tensor[index][:, :int(self.Ptrain_tensor.shape[2] / 2)] # B, T, N
        data_y = self.ytrain_tensor[index]
        time = self.Ptrain_time_tensor[index]
        real_time = self.Ptrain_final[index]['length'] 
        
        demo_list = []
        for d in range(len(self.Ptrain_final[index]['extended_static'])):
            demo_list.append(self.Ptrain_final[index]['extended_static'][d])
        demo = np.array(demo_list)
        
        return data_x, data_y, time, real_time, demo
        
        
    def __len__(self):
        return self.tot_len
    


class P19Visualization(Dataset):
    def __init__(self, idx_dataset, 
                 root_path = '/home/DAHS2/Timellm/Replicate_for_P19/',
                 PT_dict_path='dataset/data/processed_data/PT_dict_list_6.npy', 
                 outcomes_path = 'dataset/data/processed_data/arr_outcomes_6.npy'):

        # init
        self.split_id = idx_dataset #sample id set [Train, Valid, Test]
        self.seq_len = 60
        self.root_path = root_path
        self.outcomes_path = outcomes_path
        self.PT_dict_path = PT_dict_path
        self.tot_len = self.__read_data__()
        self.slide_unit = 1

        # self.tot_len = 100

    def __read_data__(self):
        
        Pdict_list = np.load(self.root_path + self.PT_dict_path, allow_pickle=True)
        arr_outcomes = np.load(self.root_path + self.outcomes_path, allow_pickle=True)
        
        self.Pdata = Pdict_list[self.split_id]
        self.Pdata_label = arr_outcomes[self.split_id]
        
        # length가 40 이상인 샘플들만 선택
        valid_indices = [i for i in range(len(self.Pdata)) if self.Pdata[i]['length'] >= 40]

        # 유효한 샘플들로 데이터와 라벨을 필터링
        self.Pdata = self.Pdata[valid_indices]
        self.Pdata_label = self.Pdata_label[valid_indices]

        # 라벨 0과 라벨 1의 인덱스를 추출
        label_0_indices = np.where(self.Pdata_label == 0)[0]
        label_1_indices = np.where(self.Pdata_label == 1)[0]

        # 각각의 샘플 개수와 max_samples 중 작은 값을 선택
        num_label_0 = min(len(label_0_indices), 1000)
        num_label_1 = min(len(label_1_indices), 1000)

        # 라벨 0과 1에서 각각 최대 num_label_0, num_label_1 개수만 추출
        selected_label_0_indices = label_0_indices[:num_label_0]
        selected_label_1_indices = label_1_indices[:num_label_1]

        # 최종 데이터셋 구성
        selected_indices = np.concatenate((selected_label_0_indices, selected_label_1_indices))
        # np.random.shuffle(selected_indices)

        self.Ptrain_final = self.Pdata[selected_indices]
        self.Ptrain_label_final = self.Pdata_label[selected_indices]

        T, F = self.Ptrain_final[0]['arr'].shape
        D = len(self.Ptrain_final[0]['extended_static'])

        Ptrain_tensor = np.zeros((len(self.Ptrain_final), T, F))
        Ptrain_static_tensor = np.zeros((len(self.Ptrain_final), D))

        for i in range(len(self.Ptrain_final)):
            Ptrain_tensor[i] = self.Ptrain_final[i]['arr']
            Ptrain_static_tensor[i] = self.Ptrain_final[i]['extended_static']

        mf, stdf = getStats(Ptrain_tensor)
        ms, ss = getStats_static(Ptrain_static_tensor, dataset='P19')

        self.Ptrain_tensor, self.Ptrain_static_tensor, self.Ptrain_time_tensor, self.ytrain_tensor = tensorize_normalize(
            self.Ptrain_final, self.Ptrain_label_final, mf, stdf, ms, ss)
        
        self.tot_len = len(self.Ptrain_tensor)
        
        return self.tot_len

    def __getitem__(self, index):
        
        data_x = self.Ptrain_tensor[index][:, :int(self.Ptrain_tensor.shape[2] / 2)] # B, T, N
        data_y = self.ytrain_tensor[index]
        time = self.Ptrain_time_tensor[index]
        real_time = self.Pdata[index]['length'] 
        
        demo_list = []
        for d in range(len(self.Pdata[index]['extended_static'])):
            demo_list.append(self.Pdata[index]['extended_static'][d])
        demo = np.array(demo_list)
        
        return data_x, data_y, time, real_time, demo
        
        
    def __len__(self):
        return self.tot_len


def getStats(P_tensor):
    N, T, F = P_tensor.shape
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    mf = np.zeros((F, 1))
    stdf = np.ones((F, 1))
    eps = 1e-7
    for f in range(F):
        vals_f = Pf[f, :]
        vals_f = vals_f[vals_f > 0]
        mf[f] = np.mean(vals_f)
        stdf[f] = np.std(vals_f)
        stdf[f] = np.maximum(stdf[f], eps)
    return mf, stdf

def getStats_static(P_tensor, dataset='P12'):
    N, S = P_tensor.shape
    Ps = P_tensor.transpose((1, 0))
    ms = np.zeros((S, 1))
    ss = np.ones((S, 1))

    if dataset == 'P12':
        # ['Age' 'Gender=0' 'Gender=1' 'Height' 'ICUType=1' 'ICUType=2' 'ICUType=3' 'ICUType=4' 'Weight']
        bool_categorical = [0, 1, 1, 0, 1, 1, 1, 1, 0]
    elif dataset == 'P19':
        # ['Age' 'Gender' 'Unit1' 'Unit2' 'HospAdmTime' 'ICULOS']
        bool_categorical = [0, 1, 0, 0, 0, 0]
    elif dataset == 'eICU':
        # ['apacheadmissiondx' 'ethnicity' 'gender' 'admissionheight' 'admissionweight'] -> 399 dimensions
        bool_categorical = [1] * 397 + [0] * 2

    for s in range(S):
        if bool_categorical == 0:  # if not categorical
            vals_s = Ps[s, :]
            vals_s = vals_s[vals_s > 0]
            ms[s] = np.mean(vals_s)
            ss[s] = np.std(vals_s)
    return ms, ss

def mask_normalize_static(P_tensor, ms, ss):
    N, S = P_tensor.shape
    Ps = P_tensor.transpose((1, 0))

    # input normalization
    for s in range(S):
        Ps[s] = (Ps[s] - ms[s]) / (ss[s] + 1e-18)

    # set missing values to zero after normalization
    for s in range(S):
        idx_missing = np.where(Ps[s, :] <= 0)
        Ps[s, idx_missing] = 0

    # reshape back
    Pnorm_tensor = Ps.reshape((S, N)).transpose((1, 0))
    return Pnorm_tensor

def mask_normalize(P_tensor, mf, stdf, lengths):
    """ Normalize time series variables. Missing ones are set to zero after normalization. """
    N, T, F = P_tensor.shape

    # Initialize the mask based on lengths
    M = np.zeros((N, T, F))
    for i in range(N):
        length = lengths[i]  # 실제 체류한 길이
        M[i, :length, :] = 1  # 체류한 길이까지는 1로 설정

    # Normalize the time series variables
    Pf = P_tensor.transpose((2, 0, 1)).reshape(F, -1)
    M_3D = M.transpose((2, 0, 1)).reshape(F, -1)
    for f in range(F):
        Pf[f] = (Pf[f] - mf[f]) / (stdf[f] + 1e-18)
    Pf = Pf * M_3D  # Only apply normalization where M is 1 (valid data)

    # Reshape Pf back to original dimensions and add the mask M as the last feature
    Pnorm_tensor = Pf.reshape((F, N, T)).transpose((1, 2, 0))
    Pfinal_tensor = np.concatenate([Pnorm_tensor, M], axis=2)
    
    return Pfinal_tensor

def tensorize_normalize(P, y, mf, stdf, ms, ss):
    T, F = P[0]['arr'].shape
    D = len(P[0]['extended_static'])

    P_tensor = np.zeros((len(P), T, F))
    P_time = np.zeros((len(P), T, 1))
    P_static_tensor = np.zeros((len(P), D))
    lengths = np.zeros(len(P), dtype=int)  # 각 시퀀스의 실제 체류 길이를 저장할 배열

    for i in range(len(P)):
        P_tensor[i] = P[i]['arr']
        P_time[i] = P[i]['time']
        P_static_tensor[i] = P[i]['extended_static']
        lengths[i] = P[i]['length']  # 실제 체류한 길이를 저장

    P_tensor = mask_normalize(P_tensor, mf, stdf, lengths)  # lengths를 함께 전달
    P_tensor = torch.Tensor(P_tensor)

    P_time = torch.Tensor(P_time) / 60.0  # convert mins to hours
    P_static_tensor = mask_normalize_static(P_static_tensor, ms, ss)
    P_static_tensor = torch.Tensor(P_static_tensor)

    y_tensor = torch.Tensor(y[:, 0]).type(torch.LongTensor)
    return P_tensor, P_static_tensor, P_time, y_tensor

