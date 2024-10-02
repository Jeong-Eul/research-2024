from math import sqrt
import pdb
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel
import transformers
from utils.tools import load_content, load_vocabulary, load_domain_content, load_variable_content, load_missing_token, min_with_na_handling, max_with_na_handling, median_with_na_handling
from layers.Embed import Patching
import torch.nn.functional as F
transformers.logging.set_verbosity_error()

import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, d_ff, target_window, head_dropout=0):
        super().__init__()

        self.window = target_window
        self.linear_1 = nn.Linear(d_ff, d_ff)
        self.linear_2 = nn.Linear(d_ff, target_window)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(head_dropout)
        
        nn.init.xavier_uniform_(self.linear_1.weight)

    def forward(self, x):
        out = self.linear_1(x)
        out = self.relu(out)
        out = self.linear_2(out)
        
        return out
    
class LabEmbedding(nn.Module):
    def __init__(self, num_variables, d_ff, B):
        super(LabEmbedding, self).__init__()
        
        self.B = B
        self.num_variables = num_variables
        self.weights = nn.Parameter(torch.randn(num_variables, d_ff))
        self.biases = nn.Parameter(torch.randn(num_variables, d_ff))
        
        self.d_ff = d_ff
        
    def forward(self, measurement_times, measurement_values):
        """
        measurement_times: list of tensors of shape [Ti]
        measurement_values: list of tensors of shape [Ti]
        
        Returns:
        embedding_vectors: tensor of shape [N, d_ff]
        """
        N = len(measurement_times)
        embedding_vectors = torch.zeros(N, self.d_ff)
        device = self.weights.device
        
        for n in range(N):
            times = measurement_times[n]
            values = measurement_values[n]
            
            if times.numel() == 0:  # 측정 시간이 없으면 해당 변수는 0으로 처리
                times = torch.tensor([0], device=device)  # 디바이스 일관성 유지
                values = torch.tensor([0], device=device)

            W_n = self.weights[n]
            b_n = self.biases[n]
            
            value_embeddings = values.unsqueeze(-1) * W_n + b_n  # [Ti, d_ff]

            if torch.sum(times) == 0:  # 시간이 0인 경우 안전한 처리
                weights = torch.zeros_like(times, device=device)  # 가중치를 모두 0으로 설정
            else:
                weights = times / torch.sum(times)  # [Ti]

            weighted_sum = torch.sum(weights.unsqueeze(-1) * value_embeddings, dim=0)  # [d_ff]
            
            embedding_vectors[n] = weighted_sum
        
        return embedding_vectors.view(self.B, int(self.num_variables/self.B), -1)
    
def compute_frequencies(measured_times, real_time, B, N):
    frequencies = []
    for b in range(B):
        for n in range(N):
            times = measured_times[b * N + n]
            freq = len(times) / real_time[b]
            frequencies.append(freq.float())
    return frequencies

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.d_llm = configs.llm_dim
        self.vocab = configs.vocab
        self.d_model = configs.d_model
        self.vital_index = configs.vital_index
        self.lab_index = configs.lab_index
        self.n_heads = configs.n_heads
        self.enc_in = configs.enc_in
        self.d_llm = configs.llm_dim
        self.n_stats = 4

        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained("huggyllama/llama-7b")
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            # self.llama_config.hidden_size = 512
            try:
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    "huggyllama/llama-7b",
                    trust_remote_code=True,
                    # local_files_only=True,
                    config=self.llama_config,
                    ignore_mismatched_sizes=True
                    # load_in_4bit=True,
                )
                # self.quantized_llm = torch.quantization.quantize_dynamic(
                #     self.llm_model, {torch.nn.Linear}, dtype=torch.qint8
                # )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    "huggyllama/llama-7b",
                    trust_remote_code=True,
                    # local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    "huggyllama/llama-7b",
                    trust_remote_code=True,
                    # local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    "huggyllama/llama-7b",
                    trust_remote_code=True,
                    # local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    "openai-community/gpt2",
                    trust_remote_code=True,
                    # local_files_only=True,
                    config=self.gpt2_config,
                    # load_in_4bit=True,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    "openai-community/gpt2",
                    trust_remote_code=True,
                    # local_files_only=False,
                    config=self.gpt2_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    "openai-community/gpt2",
                    trust_remote_code=True,
                    # local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    "openai-community/gpt2",
                    trust_remote_code=True,
                    # local_files_only=False
                )
                
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = 'Sepsis'
        else:
            self.description = 'Sepsis'

        self.dropout = nn.Dropout(configs.dropout)
        
        # Vital mapping
        self.vital_mapping = nn.Linear(self.d_llm, self.d_ff)
        
        # # Lab_mapping
        self.lab_embedding = LabEmbedding(configs.batch_size*len(self.lab_index), self.d_ff, configs.batch_size)
        
        # Reprogramming
        self.reprogramming_layer = ReprogrammingLayer(self.d_ff, self.n_heads, d_keys = self.d_ff, d_llm=self.d_llm)
        
        # Text Prototype
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        
        # Variable aggragation
        self.output_projection = ClassificationHead(self.d_ff, 1, head_dropout=0.3)
        self.variable_attn = Variable_Attention(self.d_ff, self.n_heads, configs.batch_size, self.enc_in, d_keys = self.d_ff, attention_dropout=0.1)
    
    def forward(self, x_enc, time, real_time, demo, null_mask, mask=None):
        dec_out = self.forecast(x_enc, time, real_time, demo, null_mask)
        return dec_out #B, pred_window

    def forecast(self, x_enc, time, real_time, demo, null_mask):

        x_vital = x_enc[:, :, self.vital_index] * null_mask[:, :, self.vital_index]
        N_vital = x_vital.size()[-1]
        
        x_lab = x_enc[:, :, self.lab_index] * null_mask[:, :, self.lab_index]
        N_lab = x_lab.size()[-1]
        nan_mask = torch.isnan(x_lab)
        x_lab = torch.where(nan_mask, torch.tensor(0.0), x_lab)    
        
        # prompt as prefix

        B, T, N = x_vital.size()

        domain_dcb = load_domain_content('Sepsis')

        prompts = []
        for b in range(B):
            batch_data = x_vital[b]  # (T, N)
            batch_time = time[b]
            
            real_time_int = real_time[b].item()
            batch_demo_inf = demo[b]
            age = str(batch_demo_inf[0].item())
            sex = 'Male' if batch_demo_inf[1].item() == 1 else 'Female'
            
            if batch_demo_inf[2].item() > 0:
                icu = "medical intensive care unit"
            elif batch_demo_inf[3].item() > 0:
                icu = "surgical intensive care unit"
            else:
                icu = "intensive care unit"

            for n in range(N):
                valid_indices = ~torch.isnan(batch_data[:real_time[b]][:, n])
                valid_tensor = batch_data[:real_time[b]][:, n][valid_indices]

                # Find missing time points
                missing_indices = np.where(np.isnan(batch_data[:real_time[b]][:, n].cpu().numpy()))[0]
                missing_time_points = ', '.join([f't_{i}' for i in missing_indices]) if len(missing_indices) > 0 else 'none'
                
                try:
                    min_values = str(torch.min(valid_tensor, dim=0).values.item())
                except:
                    min_values = 'Not measured'
                
                try:
                    max_values = str(torch.max(valid_tensor, dim=0).values.item())
                except:
                    max_values = 'Not measured'
                
                try:
                    median = str(torch.median(valid_tensor, dim=0).values.item())
                except:
                    median = 'Not measured'
                
                try:
                    trd = valid_tensor.diff(dim=0).sum(dim=0).values.item()
                except:
                    trd = 'Not measured'
                
                if isinstance(trd, (int, float)) and trd > 0:
                    result = 'upward'
                elif trd == 'Not measured':
                    result = 'unknown'
                else:
                    result = 'downward'

                prompt_batch = (
                    f"<|start_prompt|>Who you are: You're an ICU doctor. You didn't take a measurement at a certain time during a patient's vital signs measurement. But you can estimate the value. "
                    f"Task Description: Make a time series with a given missing value into a fully observed time series embedding.;"
                    f"This patient is {sex}, {age} years old, and went to the {icu}, {str(round(batch_demo_inf[4].item()*-1))} hours after hospital admit and had stayed there for {str(real_time_int)} hours. "
                    "Input statistics: "
                    f"min value {min_values}, " 
                    f"max value {max_values}, "
                    f"median value {median}, "
                    f"the trend of input is {result}. "
                    f"Missing values occurred at time: {missing_time_points}.<|end_prompt|>"
                )
                prompts.append(prompt_batch)
        
                
        prompt = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False, max_length=1024).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (B*N_vital, prompt_token, llm_dim)

        # Text Prototype
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        
        # Vital Reprogramming
        rep_out = self.reprogramming_layer(x_vital, source_embeddings, source_embeddings)
        
        # Concatenate
        total_out = torch.cat((prompt_embeddings, rep_out), dim=1) # B*N_vital, prompt + Time, d_llm
        
        # Make attention mask for LLM
        nan_mask = null_mask[:, :, :N_vital] == 0
        processed_tensor = torch.where(nan_mask, torch.tensor(0.0), null_mask[:, :, :N_vital])
        processed_tensor = torch.nan_to_num(processed_tensor, nan=1)
        ts_mask = processed_tensor.permute(0, 2, 1).reshape(B * N_vital, T)
        
        prompt_length = prompt_embeddings.shape[1]
        attention_mask = torch.ones(B*N_vital, prompt_length+T)
        attention_mask[:, -T:] = ts_mask

        # LLM Body
        out = self.llm_model(inputs_embeds=total_out, attention_mask=attention_mask.to(x_enc.device)).hidden_states[-1] # B*N_vital, prompt + Time, d_llm

        # Truncate
        out = out[:, -T:]
        out = torch.sum(out, dim=1)/sqrt(self.d_llm) # B*N_vital, d_llm
        
        # Lab embedding
        measured_times = []
        measured_values = [] 

        for b in range(B):
            for n in range(N_lab):
                indices = torch.nonzero(x_lab[b, :, n], as_tuple=True)[0]  # [T_indices]
                values = x_lab[b, indices, n]  # [T_indices]
                
                measured_times.append(indices)  # 측정된 시간
                measured_values.append(values)  # 측정된 값
        
        embeded_lab = self.lab_embedding(measured_times, measured_values)
        frequencies = compute_frequencies(measured_times, real_time, B, N_lab)
        weight_frequency = torch.stack(frequencies)
        
        lab_processed = embeded_lab.to(x_enc.device) * weight_frequency.view(B, N_lab, 1).to(x_enc.device) # 더 많이 측정할 수록 값이 강조될 것임
        
        # Vital embedding
        maped_vital = self.vital_mapping(out.view(-1, self.d_llm)) # B*N_vital, D
        vital_processed = maped_vital.view(B, -1, self.d_ff)
        
        # Total processing 
        total = torch.cat([vital_processed, lab_processed], dim = 1) # B, N, D
        output = self.variable_attn(total) # B, d_ff
        
        # Classification Head
        dec_out = self.output_projection(output) #B, target window
        
        return dec_out

class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.01):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        self.d_keys = d_keys
        self.n_heads = n_heads
        
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm) # mask 된 부분은 아예 무시 되도록 bias 계산 x
        
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, time_series, source_embedding, value_embedding):
        
        B, T, N = time_series.shape # B, T, N
        S, _ = source_embedding.shape
        H = self.n_heads
        
        # Query
        time_series = time_series.permute(0, 2, 1).reshape(B*N, T)
        repeated_ts = time_series.unsqueeze(-1).repeat(1, 1, self.d_keys * self.n_heads)
        
        nan_mask = torch.isnan(repeated_ts)
        Q = torch.where(nan_mask, torch.tensor(0.0), repeated_ts).view(B*N, T, H, -1) #B*N, T, H, d_ff
        
        # Key
        K = self.key_projection(source_embedding).view(S, H, -1)
        
        # Value
        V = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(Q, K, V) # B_N, T, H, E

        out = out.reshape(B*N, T, -1)

        return self.dropout(self.out_projection(out))

    def reprogramming(self, Q, K, V):
        B_N, T, H, d_ff = Q.shape
        
        scale = 1. / sqrt(d_ff)
        scores = torch.einsum("bthe,she->bhts", Q, K)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhts,she->bthe", A, V)
    
        return reprogramming_embedding
    
    
class Variable_Attention(nn.Module):
    def __init__(self, D, n_heads, B, N, d_keys=None, attention_dropout=0.1):
        super(Variable_Attention, self).__init__()

        d_keys = d_keys or (D // n_heads)
        self.d_keys = d_keys
        self.B = B
        self.N = N
        self.query_projection = nn.Linear(D, d_keys * n_heads, bias=True)
        self.key_projection = nn.Linear(D, d_keys * n_heads, bias=True)
        self.value_projection = nn.Linear(D, d_keys * n_heads, bias=True)
        self.summary_projection = nn.Linear(N, 1)
        self.out_projection_layer = nn.Linear(D * n_heads, D, bias = True)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding):
        B, N, D = target_embedding.shape # B, N, D
        
        H = self.n_heads
        Q = self.query_projection(target_embedding).view(self.B, self.N, H, self.d_keys) # B, N, H, E
        K = self.key_projection(target_embedding).view(self.B, self.N, H, self.d_keys) # B, N, H, E
        V = self.value_projection(target_embedding).view(self.B, self.N, H, self.d_keys) # B, N, H, E
  
        beta = self.get_attention(Q, K) # B, H, N, 1
        beta_ori = beta.squeeze(-1) # B, H, N
        beta_cal = beta_ori.permute(0, 2, 1) # B, N, H
        
        weighted_tensor = V * beta_cal.unsqueeze(-1)
        result = weighted_tensor.sum(dim=1)

        out = self.out_projection_layer(result.view(B, -1))
        return self.dropout(out)
    
    def get_attention(self, Q, K):
        
        B, N, H, E = Q.shape
        B, N, H, E = K.shape

        scale = 1. / torch.sqrt(torch.tensor(E))

        scores = torch.einsum("bnhd,bmhd->bhnm", Q, K) # B, H, N, N
        beta = self.dropout(torch.softmax(self.summary_projection(scale * scores), dim=-2)) # B, H, N, 1
        return beta

class Temporal_Attention(nn.Module):
    def __init__(self, D, n_heads, B, T, d_keys=None, attention_dropout=0.1):
        super(Temporal_Attention, self).__init__()

        d_keys = d_keys or (D // n_heads)
        self.d_keys = d_keys
        self.B_N = B
        self.T = T
        self.query_projection = nn.Linear(D, d_keys * n_heads, bias=True)
        self.key_projection = nn.Linear(D, d_keys * n_heads, bias=True)
        self.value_projection = nn.Linear(D, d_keys * n_heads, bias=True)
        self.summary_projection = nn.Linear(T, 1)
        self.out_projection_layer = nn.Linear(D * n_heads, D, bias = True)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding):
        B_N, T, D = target_embedding.shape # B_N, T, D
        
        H = self.n_heads
        Q = self.query_projection(target_embedding).view(self.B_N, self.T, H, self.d_keys) # B_N, T, H, E
        K = self.key_projection(target_embedding).view(self.B_N, self.T, H, self.d_keys) # B_N, T, H, E
        V = self.value_projection(target_embedding).view(self.B_N, self.T, H, self.d_keys) # B_N, T, H, E
  
        beta = self.get_attention(Q, K) # B_N, H, T, 1
        beta_ori = beta.squeeze(-1) # B_N, H, T
        beta_cal = beta_ori.permute(0, 2, 1).unsqueeze(-1) # B_N, T, H, 1
        
        weighted_tensor = V * beta_cal
        result = weighted_tensor.sum(dim=1) # B_N, H, E

        out = self.out_projection_layer(result.view(B_N, -1)) # B_N, E
        return self.dropout(out)
    
    def get_attention(self, Q, K):
        
        B_N, T, H, E = Q.shape
        B_N, T, H, E = K.shape

        scale = 1. / torch.sqrt(torch.tensor(E))

        scores = torch.einsum("bthd,bjhd->bhtj", Q, K) # B, H, T, T
        beta = self.dropout(torch.softmax(self.summary_projection(scale * scores), dim=-2)) # B, H, T, 1
        return beta
