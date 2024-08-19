import argparse
import torch
import torch.nn as nn
from accelerate import Accelerator, DeepSpeedPlugin
import deepspeed
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

import sys
module_path='/home/DAHS2/Timellm/Replicate_for_P19'
if module_path not in sys.path:
    sys.path.append(module_path)
    
from model import TimeLLM, EHRTime_LLM, EHRTime_LLM_v2
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
import os
import gc

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/x86_64-linux-gnu:' + os.environ.get('LD_LIBRARY_PATH', '')

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
    parser.add_argument('--mode', type=str, default='Train',
                        help='Processing options, options:[Train, Valid, Test]')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='P19', help='dataset type')
    parser.add_argument('--root_path', type=str, default='/home/DAHS2/Timellm/Replicate_for_P19/', help='root path of the data file')
    parser.add_argument('--PT_dict_path', type=str, default='dataset/data/processed_data/PT_dict_list_6.npy', help='dataset-timeseries file')
    parser.add_argument('--outcomes_path', type=str, default='dataset/data/processed_data/arr_outcomes_6.npy', help='dataset-outcomes file')
    parser.add_argument('--split_num', type=int, default=1, help='cross validation set')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--undersampling', action='store_true', help='whether random under sampling', default=False)
    parser.add_argument('--upsampling', action='store_true', help='whether random over sampling', default=True)

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
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='BCE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=True)
    parser.add_argument('--llm_layers', type=int, default=6)

    args = parser.parse_args()
    
    args.vocab = load_vocabulary()
    args.domain = load_domain_content('Sepsis')
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], gradient_accumulation_steps=2, mixed_precision="fp16", deepspeed_plugin=None)
    
    def create_dir_if_not_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    checkpoint_dir = './model_checkpoint'
    create_dir_if_not_exists(checkpoint_dir)
    
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_{}_pl{}_dm{}_nh{}_df{}_il_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.d_ff,
            ii)
        print('Data loading......')
        
        train_data, train_loader = P19_DataLoader(args, args.data, flag = 'Train')
        vali_data, vali_loader = P19_DataLoader(args, args.data, flag = 'Valid')

        if args.model == 'TimeLLM':
            model = TimeLLM.Model(args)
        elif args.model =='EHRTimeLLM':
            model = EHRTime_LLM_v2.Model(args).float()

        path = os.path.join(args.checkpoints,
                            setting + '-' + args.model_comment)  # unique checkpoint saving path
        
        if not os.path.exists(path) and accelerator.is_local_main_process:
            os.makedirs(path)
        
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

        trained_parameters = []
        for p in model.parameters():
            p.data = p.data.float()
            if p.requires_grad is True:
                trained_parameters.append(p)

        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

        if args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=30, eta_min=1e-9)
        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=args.pct_start,
                                                epochs=args.train_epochs,
                                                max_lr=args.learning_rate)

        criterion = nn.BCEWithLogitsLoss()
    
        train_loader, vali_loader, model, model_optim, scheduler = accelerator.prepare(
            train_loader, vali_loader, model, model_optim, scheduler)

        # if args.use_amp:
        #     scaler = torch.amp.GradScaler("cuda")
            
        best_loss = 0.7
        # torch.distributed.set_debug_level(3)
        for epoch in range(args.train_epochs):
            iter_count = 0
            train_loss = []

            model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_time, batch_real_time, batch_demo) in tqdm(enumerate(train_loader)):
                with accelerator.accumulate(model):
                    iter_count += 1
                    # model_optim.zero_grad()

                    batch_x = batch_x.float().to(accelerator.device)
                    batch_y = batch_y.float().to(accelerator.device).unsqueeze(1)
                    batch_time = batch_time.float().to(accelerator.device)
            
                    if args.use_amp:
                        with accelerator.autocast():
                            if args.output_attention:
                                outputs = model(batch_x, batch_time, batch_real_time, batch_demo)[0]
                            else:
                                outputs = model(batch_x, batch_time, batch_real_time, batch_demo)

                            loss = criterion(outputs, batch_y.to(accelerator.device))
                            train_loss.append(loss.item())
                            
                        accelerator.backward(loss)
                        model_optim.step()
                        model_optim.zero_grad()  # batch 2개에 대한 gradient 누적을 업데이트
             
                        
                    else:
                        if args.output_attention:
                            outputs = model(batch_x, batch_time, batch_real_time, batch_demo)[0]
                        else:
                            outputs = model(batch_x, batch_time, batch_real_time, batch_demo)

                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                        
                        accelerator.backward(loss)
                        model_optim.step()
                        model_optim.zero_grad()
                        
                        
                    # if (i + 1) % (epoch+1) == 0:
                    #     accelerator.print(
                    #         "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    #     speed = (time.time() - time_now) / iter_count
                    #     left_time = speed * ((args.train_epochs - epoch - 1) * train_steps)
                    #     accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    #     iter_count = 0
                    #     time_now = time.time()
    
            accelerator.wait_for_everyone()    
            accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = vali(args, accelerator, model, vali_data, vali_loader, criterion)
            # test_loss = vali(args, accelerator, model, test_data, test_loader, criterion)
            accelerator.print(
                "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} ".format(
                    epoch + 1, train_loss, vali_loss))
            
            early_stopping(vali_loss, model, path)
            if early_stopping.early_stop:
                accelerator.print("Early stopping")
                break

            if args.lradj != 'TST':
                if args.lradj == 'COS':
                    scheduler.step()
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                else:
                    if epoch == 0:
                        args.learning_rate = model_optim.param_groups[0]['lr']
                        accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

            else:
                accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
            
        accelerator.wait_for_everyone()        
        if accelerator.is_main_process:
            model_to_save = accelerator.unwrap_model(model)
            save_path = os.path.join(checkpoint_dir, f"model_full_epoch{epoch + 1}_trn_loss{train_loss}_vld_loss{vali_loss}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': model_optim.state_dict(),
                'loss': vali_loss,
            }, save_path)
            accelerator.print(f"Model saved to {save_path}")
        gc.collect()
        torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    # if accelerator.is_local_main_process:
    #     path = './checkpoints'  # unique checkpoint saving path
    #     del_files(path)  # delete checkpoint files
    #     accelerator.print('success delete checkpoints')
    accelerator.free_memory()
