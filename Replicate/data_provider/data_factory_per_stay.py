from torch.utils.data import DataLoader
from data_provider.data_loader_per_stay import PUnitSequenceDataset 
import numpy as np
import torch

data_dict = {
    'hirid': PUnitSequenceDataset,
}

def custom_collate_fn(batch):
    seq_x, seq_y, time = zip(*batch)
    return torch.stack(seq_x), torch.stack(seq_y), torch.stack(time)

def PUnitSequence_Provider(args, data, flag='train'):
    Data = data_dict[data]

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        p_path = args.tst_split_path
        
    elif flag == 'valid':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        p_path = args.vld_split_path
    else:
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        p_path = args.trn_split_path
        

    data_set = Data(
        root_path=args.root_path,
        data_path=p_path,
        stay_id = args.stay_id,
        target=args.target,
        percent=args.percent,
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        )
    # collate_fn=custom_collate_fn
    return data_set, data_loader