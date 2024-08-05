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