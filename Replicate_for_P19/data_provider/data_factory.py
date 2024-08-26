from torch.utils.data import DataLoader
from data_provider.data_loader import P19Datasaets, P19DatasaetsDownSampled, P19DatasaetsUpSampled, P19Visualization
import numpy as np



data_dict = {
    'P19': P19Datasaets,
}

def P19_DataLoader(args, data, flag='Train'):
    
    Data = data_dict[data]
    
    split_idx = args.split_num # set this parameter as argment parameter

    split_path = args.root_path + 'dataset/data/splits/phy19_split' + str(split_idx+1) + '_new.npy'
    idx_train, idx_val, idx_test = np.load(split_path, allow_pickle=True)
    
    if flag == 'Test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        splited_dataset = idx_test
        
    elif flag == 'Valid':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        splited_dataset = idx_val
        
    elif flag == 'Test inference':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        splited_dataset = idx_test
        
    elif flag == 'Valid inference':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        splited_dataset = idx_val
    
    elif flag == 'Visualization':
        Data = P19Visualization
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        splited_dataset = idx_train
        
    else:
        if args.undersampling:
            Data = P19DatasaetsDownSampled
        elif args.upsampling:
            Data = P19DatasaetsUpSampled
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        splited_dataset = idx_train
        

    data_set = Data(
        idx_dataset = splited_dataset,
        root_path = args.root_path,
        PT_dict_path = args.PT_dict_path,
        outcomes_path = args.outcomes_path,
    )

  
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        )
    
    return data_set, data_loader