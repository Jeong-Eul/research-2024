import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

# Dataloader that processing patient unit stay independently
class PUnitSequenceDataset(Dataset):
    def __init__(self, root_path, size=None, data_path='HiRID_shock_10min.csv.gz', stay_id = 'patientid',
                 target='shock_next_6h', percent=100):
        if size == None:
            self.seq_len = 10
            self.label_len = 1
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        self.set_type = 0
        self.stay_id = stay_id
        self.percent = percent
        self.target = target
        self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.slide_unit = 1
        self.tot_len = sum((len(pat_data) - self.seq_len - self.pred_len + 1) // self.slide_unit 
                           for pat_data in self.patient_data.values())

    def __read_data__(self):
        df_raw = pd.read_csv(self.root_path + '/' + self.data_path)
        df_raw.drop(['Vasopressor', 'Annotation'], axis = 1, inplace=True)
        self.patient_data = {}
        
        for stay_id, patient_df in df_raw.groupby(self.stay_id):
            self.patient_data[stay_id] = patient_df

        self.data_x = df_raw.drop(columns=[self.target]).values
        self.data_y = df_raw[self.target].values

    def __getitem__(self, index):
        current_index = 0
        for _, patient_df in self.patient_data.items():
            patient_len = len(patient_df)
            num_sequences = (patient_len - self.seq_len - self.pred_len + 1) // self.slide_unit
            if index < current_index + num_sequences:
                patient_index = index - current_index
                s_begin = (patient_index * self.slide_unit)
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len

                seq_x = patient_df.iloc[s_begin:s_end].drop(columns=[self.target, self.stay_id]).values
                seq_y = patient_df.iloc[s_end:r_end][self.target].values
                time = patient_df.iloc[s_begin:s_end]['Time_since_ICU_admission'].values

                return seq_x, seq_y, time

            current_index += num_sequences

        raise IndexError('Index out of range')

    def __len__(self):
        return self.tot_len
    
class PUnitSequenceDataset_V2(Dataset):
    def __init__(self, root_path, size=None, data_path='HiRID_shock_10min.csv.gz', stay_id = 'patientid',
                 target='shock_next_6h', percent=100):
        if size == None:
            self.seq_len = 10
            self.label_len = 1
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        self.set_type = 0
        self.stay_id = stay_id
        self.percent = percent
        self.target = target
        self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.slide_unit = 1
        self.tot_len = sum((len(pat_data) - self.seq_len - self.pred_len + 1) // self.slide_unit 
                           for pat_data in self.patient_data.values())

    def __read_data__(self):
        df_raw = pd.read_csv(self.root_path + '/' + self.data_path)
        df_raw.drop(['Vasopressor', 'Annotation'], axis = 1, inplace=True)
        self.patient_data = {}
        
        for stay_id, patient_df in df_raw.groupby(self.stay_id):
            self.patient_data[stay_id] = patient_df

        self.data_x = df_raw.drop(columns=[self.target]).values
        self.data_y = df_raw[self.target].values

    def __getitem__(self, index):
        current_index = 0
        for _, patient_df in self.patient_data.items():
            patient_len = len(patient_df)
            num_sequences = (patient_len - self.seq_len - self.pred_len + 1) // self.slide_unit
            if index < current_index + num_sequences:
                patient_index = index - current_index
                s_begin = (patient_index * self.slide_unit)
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len

                seq_x = patient_df.iloc[s_begin:s_end].drop(columns=[self.target]).values
                seq_y = patient_df.iloc[s_end:r_end][self.target].values
                time = patient_df.iloc[s_begin:s_end]['Time_since_ICU_admission'].values

                return seq_x, seq_y, time

            current_index += num_sequences

        raise IndexError('Index out of range')

    def __len__(self):
        return self.tot_len