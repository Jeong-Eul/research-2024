import argparse
import os
import sys
import time
import numpy as np
import pandas as pd
import random
from sklearn.metrics import precision_score, recall_score ,f1_score, confusion_matrix, roc_auc_score, accuracy_score, classification_report
from tqdm import tqdm
from multiprocessing import freeze_support
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
import gc
import warnings

warnings.filterwarnings("ignore")
if __name__ == '__main__':
    freeze_support()

    parser = argparse.ArgumentParser(description="Train the Fluids model for threshold", 
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data
    parser.add_argument("--train_data_dir", default="/home/DAHS2/Timellm/Replicate/dataset/data/HiRID_shock_10min_trn_toy.csv.gz", type=str, dest="train_data_dir")
    parser.add_argument("--valid_data_dir", default="/home/DAHS2/Timellm/Replicate/dataset/data/HiRID_shock_10min_vld_toy.csv.gz", type=str, dest="valid_data_dir")
    parser.add_argument('--seed', default=9040, type=int , dest='seed')
    
    args = parser.parse_args()
    
    df_train = pd.read_csv(args.train_data_dir)
    df_valid = pd.read_csv(args.valid_data_dir)
    
    print("Complete Data laoding")
    
    trn_x, trn_y = df_train.drop(['Vasopressor', 'Annotation', 'patientid', 'shock_next_6h'], axis = 1), df_train['shock_next_6h']
    vld_x, vld_y = df_valid.drop(['Vasopressor', 'Annotation', 'patientid', 'shock_next_6h'], axis = 1), df_valid['shock_next_6h']
    
    print("LGBM fitting.....")
    lgbm_wrapper = LGBMClassifier(random_state = args.seed, verbose=-1, class_weight='balanced')
            
    lgbm_wrapper.fit(trn_x, trn_y)
    
    print("Inference.....")
    valid_preds = lgbm_wrapper.predict(vld_x)
    
    
    print("Start Evaluation....")
    accuracy = accuracy_score(vld_y, valid_preds)
    precision = precision_score(vld_y, valid_preds)
    recall = recall_score(vld_y, valid_preds)
    f1 = f1_score(vld_y, valid_preds)
    conf_matrix = confusion_matrix(vld_y, valid_preds)
    class_report = classification_report(vld_y, valid_preds)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-score: {f1:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('Classification Report:')
    print(class_report)