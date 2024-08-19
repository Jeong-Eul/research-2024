import argparse
import torch
import torch.nn as nn
from accelerate import Accelerator, DeepSpeedPlugin
import deepspeed
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import sys
module_path='/home/DAHS2/Timellm/Replicate_for_P19'
if module_path not in sys.path:
    sys.path.append(module_path)
    
from model import TimeLLM, EHRTime_LLM
from sklearn.metrics import precision_score, recall_score ,f1_score, confusion_matrix, roc_auc_score, accuracy_score, classification_report


import warnings
warnings.filterwarnings("ignore")

import logging

logging.getLogger("deepspeed").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


from data_provider.data_factory import P19_DataLoader
import time
import random
import numpy as np
import pandas as pd
import os
import gc

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content, load_vocabulary, load_domain_content
from multiprocessing import freeze_support
import torch.distributed as dist
import atexit
def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Process group destroyed.")
        
def initialize_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
        print(f"Process group initialized on rank {dist.get_rank()}.")
        
if __name__ == '__main__':
    
    freeze_support()
    
    from accelerate import Accelerator, DeepSpeedPlugin
    from accelerate import DistributedDataParallelKwargs
 
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100000"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    logging.getLogger("deepspeed").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    atexit.register(cleanup)
    initialize_distributed()
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='Time-LLM')

    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='testing framework', help='model id')
    parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, DLinear]')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='P19', help='dataset type')
    parser.add_argument('--root_path', type=str, default='/home/DAHS2/Timellm/Replicate_for_P19/', help='root path of the data file')
    parser.add_argument('--PT_dict_path', type=str, default='dataset/data/processed_data/PT_dict_list_6.npy', help='dataset-timeseries file')
    parser.add_argument('--outcomes_path', type=str, default='dataset/data/processed_data/arr_outcomes_6.npy', help='dataset-outcomes file')
    parser.add_argument('--trained_model_path', type=str, default='code/model_checkpoint/Best_model_epoch_9_loss-0.695763530927835.pt', help='trained model path')
    parser.add_argument('--split_num', type=int, default=1, help='cross validation set')
    parser.add_argument('--undersampling', action='store_true', help='whether random under sampling', default=True)

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=60, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=34, help='encoder input size')
    parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=1, help='num of heads')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--patch_len', type=int, default=3, help='dimension of fcn')
    parser.add_argument('--stride', type=int, default=1, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--prompt_domain', type=str, default='Sepsis', help='Task domain')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
    parser.add_argument('--llm_dim', type=int, default='768', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768

    # optimization
    parser.add_argument('--num_workers', type=int, default=2, help='data loader num workers')
    parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--llm_layers', type=int, default=6)

    args = parser.parse_args()
    
    args.vocab = load_vocabulary()
    args.domain = load_domain_content('Sepsis')
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='/home/DAHS2/Timellm/Replicate_for_P19/ds_config_zero2.json')
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)
    
    vali_data, vali_loader = P19_DataLoader(args, args.data, flag = 'Valid inference')
    test_data, test_loader = P19_DataLoader(args, args.data, flag = 'Test inference')

    if args.model == 'TimeLLM':
        model = TimeLLM.Model(args)
    elif args.model =='EHRTimeLLM':
        model = EHRTime_LLM.Model(args).float()
    
    checkpoint = torch.load(args.root_path + args.trained_model_path)
    model_state_dict = checkpoint['model_state_dict']
    model.load_state_dict(model_state_dict)

    get_prob = nn.Sigmoid()
    dummy_optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    test_loader, vali_loader, model, dummy_optimizer = accelerator.prepare(
        test_loader, vali_loader, model, dummy_optimizer)

    predictions_vali = []
    probabilities_vali = []
    labels_vali = []
    
    predictions_test = []
    probabilities_test = []
    labels_test = []
    
    model.eval()
    with torch.no_grad():
        print('Validation Inference start....')
        for i, (batch_x, batch_y, batch_time, batch_real_time, batch_demo) in tqdm(enumerate(vali_loader)):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_time = batch_time.float().to(accelerator.device)

            if args.use_amp:
                with torch.amp.autocast('cuda'):
                    if args.output_attention:
                        outputs = model(batch_x, batch_time, batch_real_time, batch_demo)[0]
                    else:
                        outputs = model(batch_x, batch_time, batch_real_time, batch_demo)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_time, batch_real_time, batch_demo)[0]
                else:
                    outputs = model(batch_x, batch_time, batch_real_time, batch_demo)

            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

            probs = get_prob(outputs)
            
            probabilities_vali.extend(probs.detach().cpu().numpy())
            predictions_vali.extend(np.where(probs.detach().cpu().numpy() > 0.5, 1, 0))
            labels_vali.extend(batch_y.detach().cpu().float().numpy())
        accelerator.wait_for_everyone()
        print('Validation Inference finish....')    
        print('Test Inference start....') 
        for i, (batch_x, batch_y, batch_time, batch_real_time, batch_demo) in tqdm(enumerate(test_loader)):
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_time = batch_time.float().to(accelerator.device)

            if args.use_amp:
                with torch.amp.autocast('cuda'):
                    if args.output_attention:
                        outputs = model(batch_x, batch_time, batch_real_time, batch_demo)[0]
                    else:
                        outputs = model(batch_x, batch_time, batch_real_time, batch_demo)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_time, batch_real_time, batch_demo)[0]
                else:
                    outputs = model(batch_x, batch_time, batch_real_time, batch_demo)

            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

            probs = get_prob(outputs)
            
            probabilities_test.extend(probs.detach().cpu().numpy())
            predictions_test.extend(np.where(probs.detach().cpu().numpy() > 0.5, 1, 0))
            labels_test.extend(batch_y.detach().cpu().float().numpy())
        accelerator.wait_for_everyone()
        print('Test Inference finish....') 
    print("Start Evaluation....")
    auprc_vali = average_precision_score(labels_vali, probabilities_vali)
    auroc_vali = roc_auc_score(labels_vali, probabilities_vali)

    auprc_test = average_precision_score(labels_test, probabilities_test)
    auroc_test = roc_auc_score(labels_test, probabilities_test)

    # Create a DataFrame to display results
    results_df = pd.DataFrame({
        "Set": ["Validation", "Test"],
        "AUPRC": [auprc_vali, auprc_test],
        "AUROC": [auroc_vali, auroc_test]
    })
    
    print(results_df)
