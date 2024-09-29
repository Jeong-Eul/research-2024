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

def compute_stats(data, mask):
    B, N, T = data.shape  # 배치, 변수, 시간 차원 크기 가져오기
    results = torch.zeros(B, N, 4)  # 모든 결과를 저장할 텐서 (B, N, 4)

    for batch in range(B):
        for var in range(N):
            valid_values = data[batch, var][mask[batch, var]]  # 실제 측정된 값들만 선택
            if valid_values.numel() == 0:  # 한 번도 측정되지 않은 경우
                results[batch, var] = torch.tensor([0.0, 0.0, 0.0, 0.0])
            else:
                min_val = valid_values.min()
                max_val = valid_values.max()
                median_val = valid_values.median()
                mean_val = valid_values.mean()
                results[batch, var] = torch.tensor([min_val, max_val, median_val, mean_val])

    return results

class ClassificationHead(nn.Module):
    def __init__(self, d_ff, target_window, head_dropout=0):
        super().__init__()

        self.window = target_window
        self.linear_1 = nn.Linear(d_ff, d_ff)
        self.linear_2 = nn.Linear(d_ff, target_window)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(head_dropout)
        self.batch_norm = nn.BatchNorm1d(d_ff)  # Batch Normalization 추가
        
        nn.init.xavier_uniform_(self.linear_1.weight)

    def forward(self, x):
        out = self.linear_1(x)
        out = self.batch_norm(out)  # Batch Normalization 적용
        h = self.relu(out)
        h = self.dropout(h)  # Dropout 적용
        residual = x + h
        out = self.linear_2(residual)
        
        return out
    
def t2v(tau, f, w, b, w0, b0):

        v1 = f(torch.matmul(tau, w) + b)
        v2 = torch.matmul(tau, w0) + b0
        return torch.cat([v2, v1], -1)
    
class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.sin
        
    def forward(self, tau):
        return t2v(tau, self.f, self.w, self.b, self.w0, self.b0)
    
class Time2Vec(nn.Module):
    def __init__(self, time_dim, hiddem_dim):
        super(Time2Vec, self).__init__()
        self.l1 = SineActivation(time_dim, hiddem_dim)
    def forward(self, x):
        x = self.l1(x)
        return x
    
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=60):
#         super(PositionalEncoding, self).__init__()
#         self.max_len = max_len
#         self.d_model = d_model
#         self._num_timescales = d_model // 2 # sin, cos 절반씩 사용

#     def getPE(self, time):
#         B = time.shape[1] # T, B, 1

#         time = time.float()

#         timescales = self.max_len ** np.linspace(0, 1, self._num_timescales) # num_timescales, 

#         times = torch.Tensor(time.cpu()) # T, B, 1
#         scaled_time = times / torch.Tensor(timescales[None, None, :]) # [None, None, :] -> 1, 1, num_timescales
#         # Use a 32-D embedding to represent a single time point
#         pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], axis=-1)  # T x B x d_model
#         pe = pe.type(torch.FloatTensor)

#         return pe

#     def forward(self, time):
#         pe = self.getPE(time)
#         # pe = pe.cuda() 이따가는 풀어야댐 
#         return pe

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
        
        # Time embedding
        self.time2vec = Time2Vec(1, self.d_ff)
        # self.ps_encoding = PositionalEncoding(self.d_ff, max_len=60)
        
        # Vital mapping
        self.vital_mapping = nn.Linear(self.d_llm, self.d_ff)
        
        # # Lab_mapping
        # self.lab_mapping = nn.Linear(self.seq_len, len(self.lab_index))
        self.lab_mapping = nn.Linear(2*self.d_ff, self.d_ff)
        self.lab_stats_mapping = nn.Linear(4, self.d_ff)
        
        # Reprogramming
        self.reprogramming_layer = ReprogrammingLayer(self.d_ff, self.n_heads, d_keys = self.d_ff, d_llm=self.d_llm)
        
        # Text Prototype
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 10000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        
        # Variable aggragation
        self.output_projection = ClassificationHead(self.d_ff, 1, head_dropout=0.3)
        self.variable_attn = Variable_Attention(self.d_ff, self.n_heads, configs.batch_size, self.enc_in, d_keys = self.d_ff, attention_dropout=0.1)
    
    def forward(self, x_enc, time, real_time, demo, null_mask, mask=None):
        dec_out = self.forecast(x_enc, time, real_time, demo, null_mask)
        return dec_out #B, pred_window

    def forecast(self, x_enc, time, real_time, demo, null_mask):

        # x_enc = torch.tensor(x_enc).to(dtype=torch.float32)
        # time = torch.tensor(time).to(dtype=torch.float32)
        # null_mask = torch.tensor(null_mask).to(dtype=torch.float32)

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
                    f"<|start_prompt|>Domain description: {domain_dcb} "
                    f"Task Description: Predict whether this patient will develop sepsis. ;"
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
        # out = self.llm_model(inputs_embeds=total_out).hidden_states[-1] # B*N_vital, prompt + Time, d_llm

        # Truncate
        out = out[:, -T:]
        out = torch.sum(out, dim=1)/sqrt(self.d_llm) # B*N_vital, d_llm
        
        # Time embedding for lab test measured time
        tiem_embeddings = self.time2vec(time.reshape(B, 1, T, 1).repeat(1, N_lab, 1, 1)).to(x_enc.device)
        
        lab_mask = (x_lab != 0)
        lab_mask = lab_mask.permute(0, 2, 1) # B, N, T
        lab_mask = lab_mask.unsqueeze(-1) # B, N, T, 1
        
        time_embedding_masked = tiem_embeddings * lab_mask # B, N, T, D
        
        #               position encoding
        # pe_output = self.ps_encoding(time.permute(1, 0, 2)).to(x_enc.device)
        # pe_output = pe_output.permute(1, 0, 2)
        # pe_repeated = pe_output.unsqueeze(1).repeat(1, N_lab, 1, 1) # B, N_lab, T, D
        
        # time_embedding_with_ps = pe_repeated * time_embedding_masked
        
        lab_processed = torch.einsum('bnt,bndt->bnd', x_lab.permute(0, 2, 1).float(), time_embedding_masked.permute(0, 1, 3, 2))
        scale = 1. / torch.sqrt(torch.tensor(lab_processed.shape[-1]))
        lab_processed = lab_processed * scale # B, N_lab, D
        
        lab = x_lab.permute(0, 2, 1) # B, N, T
        mask = lab != 0 
        stats = compute_stats(lab, mask) # B, N, n_stats
        
        lab_output = self.lab_stats_mapping(stats.to(x_enc.device)) # B, N, H 

        lab_processed = torch.cat([lab_output, lab_processed], dim = -1) # B, N, H + D
        lab_processed = self.lab_mapping(lab_processed) # B, N, D
        
        # Variable attention
        
        maped_vital = self.vital_mapping(out.view(-1, self.d_llm)) # B*N_vital, D
        vital_processed = maped_vital.view(B, -1, self.d_ff)
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
