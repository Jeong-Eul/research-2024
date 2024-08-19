import argparse
import torch
import torch.nn as nn
from accelerate import Accelerator, DeepSpeedPlugin
import deepspeed
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch.nn.functional as F
from models import Autoformer, DLinear, TimeLLM, TimeLLM_custom, Get_score

import warnings
warnings.filterwarnings("ignore")

import logging

logging.getLogger("deepspeed").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


from data_provider.data_factory import data_provider_testing
import time
import random
import numpy as np
import os
import gc

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content, load_vocabulary
from multiprocessing import freeze_support
if __name__ == '__main__':
    freeze_support()
    from accelerate import Accelerator, DeepSpeedPlugin
    from accelerate import DistributedDataParallelKwargs
    
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100000"

    logging.getLogger("deepspeed").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='Time-LLM')

    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='classification',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, DLinear]')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; '
                            'M:multivariate predict multivariate, S: univariate predict univariate, '
                            'MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--loader', type=str, default='modal', help='dataset type')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, '
                            'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                            'you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--prompt_domain', type=int, default=0, help='')
    parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
    parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768


    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=True)
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--percent', type=int, default=100)

    args = parser.parse_args()
    args.vocab = load_vocabulary()
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # # transformer_layer_classes = ["LlamaDecoderLayer"]  # 올바른 Transformer 레이어 클래스 이름을 넣어주세요
    # deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
    # # deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
    # accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)
    # # deepspeed_plugin.transformer_layer_classes = transformer_layer_classes
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print("Accelerator 설정 완료")

    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.des, ii)

        train_data, train_loader = data_provider_testing(args, args.data, flag = 'train')

        if args.model == 'Autoformer':
            model = Autoformer.Model(args).float()
        elif args.model == 'DLinear':
            model = DLinear.Model(args).float()
        elif args.model == 'TimeLLM':
            model = TimeLLM.Model(args).to(device)
            args.score = False
        elif args.model == 'TimeLLM-Custom':
            model = TimeLLM_custom.Model(args).to(device)
            args.score = False
        else:
            model = Get_score.Model(args).to(device)
            args.score = True

        path = os.path.join(args.checkpoints,
                            setting + '-' + args.model_comment)  # unique checkpoint saving path
        args.content = load_content(args)
        # if not os.path.exists(path) and accelerator.is_local_main_process:
        #     os.makedirs(path)
        
        print('Model Loading...')
        checkpoint = torch.load('/home/DAHS2/Timellm/Replicate/model_checkpoint/Customizing071015.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Complete')
        
        trained_parameters = []
        for p in model.parameters():
            if p.requires_grad is True:
                trained_parameters.append(p)
        
        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)
        
        time_now = time.time()
        
        # try:
        #     model_optim,train_loader, model = accelerator.prepare(model_optim,
        #         train_loader, model)
        #     print("accelerator.prepare 완료")
        # except Exception as e:
        #     print(f"accelerator.prepare 중 에러 발생: {e}")
        

        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        iter_count = 0
        if args.score == True:
            print('Get Score')
            model.eval()
    
            with torch.no_grad():
                epoch_time = time.time()
                total_samples = 0
                
                for i, (batch_x, batch_y, _, _) in tqdm(enumerate(train_loader)):

                    batch_x = batch_x.bfloat16().to(device)
                    batch_y = batch_y.bfloat16().to(device)

                    with torch.cuda.amp.autocast():
                        if args.output_attention:
                            _, score = model(batch_x)[0]
                        else:
                            _, score = model(batch_x)

                    score = score.detach().cpu().numpy()
                    # patch = patch.detach().cpu().numpy()
                    
                    if i == 0:
                        break
    
                np.save('/home/DAHS2/Timellm/Replicate/Result/Predicted Result/score', score)
                # np.save('/home/DAHS2/Timellm/Replicate/Result/Predicted Result/patch', patch)

                time_now = time.time()
                gc.collect()
                torch.cuda.empty_cache()
                print('Complete!!')
            
        else:
            print('Evaluation')
            model.eval()
            outputs_list = []

            with torch.no_grad():
                epoch_time = time.time()
                total_samples = 0
                
                for i, (batch_x, batch_y, _, _) in tqdm(enumerate(train_loader)):

                    batch_x = batch_x.bfloat16().to(device)
                    batch_y = batch_y.bfloat16().to(device)

                    with torch.cuda.amp.autocast():
                        if args.output_attention:
                            outputs = model(batch_x)[0]
                        else:
                            outputs = model(batch_x)

                    # 텐서로 변환하여 결과를 리스트에 추가
                    outputs_list.append(outputs.sigmoid_().detach().cpu().numpy())

                print('Evaluation Start')
                outputs_arrays = np.concatenate(outputs_list)
                np.save('/home/DAHS2/Timellm/Replicate/Result/Predicted Result/customizing_result', outputs_arrays)

                time_now = time.time()
                gc.collect()
                torch.cuda.empty_cache()