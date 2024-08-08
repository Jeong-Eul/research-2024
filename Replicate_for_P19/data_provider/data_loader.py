import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import warnings
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
        self.__read_data__()
        self.slide_unit = 1
        self.tot_len = len(self.split_id)

    def __read_data__(self):
        
        Pdict_list = np.load(self.root_path + self.PT_dict_path, allow_pickle=True)
        arr_outcomes = np.load(self.root_path + self.outcomes_path, allow_pickle=True)
        
        self.Pdata = Pdict_list[self.split_id]
        self.Pdata_label = arr_outcomes[self.split_id]

    def __getitem__(self, index):
        demo_list = []
        real_measured = self.Pdata[index]['length']
        real_data = self.Pdata[index]['arr'][:real_measured]
        padded_data = self.Pdata[index]['arr'][real_measured:]
        real_data[real_data == 0] = np.nan
        
        data_x = np.concatenate((real_data, padded_data), axis = 0)
        data_y = self.Pdata_label[index]
        time = self.Pdata[index]['time']
        for d in range(len(self.Pdata[index]['extended_static'])):
            demo_list.append(self.Pdata[index]['extended_static'][d])
        demo = np.array(demo_list)
        
        return data_x, data_y, time, real_measured, demo
        
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