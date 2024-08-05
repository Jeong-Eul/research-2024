import argparse
import torch
import torch.nn as nn
from accelerate import Accelerator, DeepSpeedPlugin
import deepspeed
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import Autoformer, DLinear, TimeLLM, TimeLLM_custom, Get_score
from sklearn.metrics import precision_score, recall_score ,f1_score, confusion_matrix, roc_auc_score, accuracy_score, classification_report


import warnings
warnings.filterwarnings("ignore")

import logging

logging.getLogger("deepspeed").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


from data_provider.data_factory_per_stay import PUnitSequence_Provider
import time
import random
import numpy as np
import os
import gc

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content, load_vocabulary, load_domain_content
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
    parser.add_argument('--mode', type=str, default='Train',
                        help='Processing options, options:[Train, Valid, Test]')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--stay_id', type=str, default='patientid', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/data', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='HiRID_shock_10min.csv.gz', help='data file')
    parser.add_argument('--trn_split_path', type=str, default='HiRID_shock_10min_trn.csv.gz', help='train data file')
    parser.add_argument('--vld_split_path', type=str, default='HiRID_shock_10min_vld.csv.gz', help='valid data file')
    parser.add_argument('--tst_split_path', type=str, default='HiRID_shock_10min_tst.csv.gz', help='test data file')
    parser.add_argument('--target', type=str, default='shock_next_6h', help='target feature in S or MS task')
    parser.add_argument('--loader', type=str, default='modal', help='dataset type')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
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
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=True)
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--percent', type=int, default=100)

    args = parser.parse_args()
    
    args.vocab = load_vocabulary()
    args.domain = load_domain_content('Shock')
    
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # transformer_layer_classes = ["LlamaDecoderLayer"]  # 올바른 Transformer 레이어 클래스 이름을 넣어주세요
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='/home/DAHS2/Timellm/Replicate/ds_config_zero2.json')
    # deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)
    # deepspeed_plugin.transformer_layer_classes = transformer_layer_classes
    
    
    def create_dir_if_not_exists(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    # 체크포인트를 저장할 디렉토리
    checkpoint_dir = './model_checkpoint'
    create_dir_if_not_exists(checkpoint_dir)

    print("Accelerator 설정 완료")

    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_eb_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.embed,
            ii)
        print('Data loading......')
        train_data, train_loader = PUnitSequence_Provider(args, args.data, flag = 'train')
        vali_data, vali_loader = PUnitSequence_Provider(args, args.data, flag = 'val')
        # test_data, test_loader = data_factory_per_stay.PUnitSequence_Provider(args.split_path, args, args.data, flag = 'test')

        if args.model == 'TimeLLM-Shock':
            model = TimeLLM_custom.Model(args)
        else:
            model = Get_score.Model(args)

        path = os.path.join(args.checkpoints,
                            setting + '-' + args.model_comment)  # unique checkpoint saving path
        args.content = load_content(args)
        if not os.path.exists(path) and accelerator.is_local_main_process:
            os.makedirs(path)
        
    
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

        trained_parameters = []
        for p in model.parameters():
            if p.requires_grad is True:
                trained_parameters.append(p)

        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

        if args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=8, eta_min=1e-5)
        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=args.pct_start,
                                                epochs=args.train_epochs,
                                                max_lr=args.learning_rate)

        criterion = nn.BCEWithLogitsLoss()
    

        # train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        #     train_loader, vali_loader, test_loader, model, model_optim, scheduler)
        
        # 데이터 로더와 모델 준비
        try:
            train_loader, vali_loader, model, model_optim, scheduler = accelerator.prepare(
                train_loader, vali_loader, model, model_optim, scheduler)
            print("accelerator.prepare 완료")
        except Exception as e:
            print(f"accelerator.prepare 중 에러 발생: {e}")
        

        if args.use_amp:
            scaler = torch.amp.GradScaler("cuda")
        best_loss = 1000
        
        if args.mode == 'Trian':
        
            for epoch in range(args.train_epochs):
                iter_count = 0
                train_loss = []

                model.train()
                epoch_time = time.time()
                for i, (batch_x, batch_y, batch_time) in tqdm(enumerate(train_loader)):
                    iter_count += 1
                    model_optim.zero_grad()

                    batch_x = batch_x.bfloat16().to(accelerator.device)
                    batch_y = batch_y.bfloat16().to(accelerator.device)
                    batch_time = batch_time.bfloat16().to(accelerator.device)
            
                    if args.use_amp:
                        with torch.amp.autocast('cuda'):
                            if args.output_attention:
                                outputs = model(batch_x, batch_time)[0]
                            else:
                                outputs = model(batch_x, batch_time)

                            loss = criterion(outputs, batch_y.to(accelerator.device))
                            train_loss.append(loss.item())
                    else:
                        if args.output_attention:
                            outputs = model(batch_x, batch_time)[0]
                        else:
                            outputs = model(batch_x, batch_time)

                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                        
                    if (i + 1) % (epoch+1) == 0:
                        accelerator.print(
                            "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((args.train_epochs - epoch - 1) * train_steps)
                        accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()
        
                    accelerator.backward(loss)
                    model_optim.step()

                    if args.lradj == 'TST':
                        adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                        scheduler.step()
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    
                
                accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                train_loss = np.average(train_loss)
                vali_loss = vali(args, accelerator, model, vali_data, vali_loader, criterion)
                # test_loss = vali(args, accelerator, model, test_data, test_loader, criterion)
                accelerator.print(
                    "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} ".format(
                        epoch + 1, train_loss, vali_loss))
                # accelerator.print(
                #     "Epoch: {0} | Train Loss: {1:.7f}".format(
                #         epoch + 1, train_loss))
                if best_loss > vali_loss:
                    print('you are going to heaven')
                    model_to_save = accelerator.unwrap_model(model)
                    save_path = os.path.join(checkpoint_dir, f"Best_model_epoch_{epoch + 1}_loss-{vali_loss}.pt")

                    accelerator.wait_for_everyone()  # 모든 프로세스가 동기화되도록 기다립니다.
                    if accelerator.is_local_main_process:  # 로컬 주 프로세스만 저장을 수행합니다.
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': model_to_save.state_dict(),
                            'optimizer_state_dict': model_optim.state_dict(),
                            'loss': train_loss,
                        }, save_path)
                        best_loss = vali_loss
                        accelerator.print(f"Model saved to {save_path}")
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
                
                    
            if accelerator.is_main_process:
                model_to_save = accelerator.unwrap_model(model)
                save_path = os.path.join(checkpoint_dir, f"Final_model_full_epoch{epoch + 1}.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': model_optim.state_dict(),
                    'loss': train_loss,
                }, save_path)
                accelerator.print(f"Model saved to {save_path}")

            accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:
                path = './checkpoints'  # unique checkpoint saving path
                del_files(path)  # delete checkpoint files
                accelerator.print('success delete checkpoints')
        
        elif args.mode == 'Valid':
            print('Validation start')
            # checkpoint = torch.load('/home/DAHS2/Timellm/Replicate/scripts/model_checkpoint/Best_model_epoch_9_loss-0.9002305708135392.pt')
            # model.load_state_dict(checkpoint)
            
            model_path = '/home/DAHS2/Timellm/Replicate/scripts/model_checkpoint/Best_model_epoch_9_loss-0.9002305708135392.pt'
            
            model.load_state_dict(torch.load(model_path))
                
            predictions = []
            labels = []
            # total_mae_loss = []
            model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_time) in tqdm(enumerate(vali_loader)):
                    batch_x = batch_x.bfloat16().to(accelerator.device)
                    batch_y = batch_y.bfloat16().to(accelerator.device)
                    batch_time = batch_time.bfloat16().to(accelerator.device)

                    if args.use_amp:
                        with torch.amp.autocast('cuda'):
                            if args.output_attention:
                                outputs = model(batch_x, batch_time)[0]
                            else:
                                outputs = model(batch_x, batch_time)

                    outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

                    predictions.extend(outputs.detach().cpu().numpy())
                    labels.extend(batch_y.detach().cpu().numpy())


            print("Start Evaluation....")
            accuracy = accuracy_score(labels, labels)
            precision = precision_score(labels, labels)
            recall = recall_score(labels, labels)
            f1 = f1_score(labels, labels)
            conf_matrix = confusion_matrix(labels, labels)
            class_report = classification_report(labels, labels)

            print(f'Accuracy: {accuracy:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'F1-score: {f1:.4f}')
            print('Confusion Matrix:')
            print(conf_matrix)
            print('Classification Report:')
            print(class_report)