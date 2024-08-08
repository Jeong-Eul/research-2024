from math import sqrt

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel
import transformers
from utils.tools import load_content, load_vocabulary, load_domain_content, load_variable_content, load_missing_token, min_with_na_handling, max_with_na_handling, median_with_na_handling

transformers.logging.set_verbosity_error()

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        #x ->  B, N, d_ff, patch num
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear_1 = nn.Linear(nf * n_vars , target_window)
        self.dropout = nn.Dropout(head_dropout)
        
        nn.init.xavier_uniform_(self.linear_1.weight)
        
    def forward(self, x):
        
        x = self.flatten(x) # B, N * (prompt_token + seq len) * d_ff
        x = self.linear_1(x)
        x = self.dropout(x)
        
        return x

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.d_llm = configs.llm_dim
        self.vocab = configs.vocab

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
            
        # for param in self.quantized_llm.parameters():
        #     param.requires_grad = False

        if configs.prompt_domain:
            self.description = 'Sepsis'
        else:
            self.description = 'Sepsis'

        self.dropout = nn.Dropout(configs.dropout)
        self.embedding_layers = self.llm_model.get_input_embeddings()
        self.reprogramming_layer = ReprogrammingLayer(configs.seq_len, configs.n_heads, None, self.d_llm)
        self.enc_in = configs.enc_in
    
    def forward(self, x_enc, time, real_time, demo, mask=None):
        dec_out = self.forecast(x_enc, time, real_time, demo)
        return dec_out #B, pred_window

    def forecast(self, x_enc, time, real_time, demo):

        x_enc = x_enc.to(dtype=torch.float32)
        time = time.to(dtype=torch.float32)
        B, T, N = x_enc.size()  # (B, 60, 34)

        variable_descripttion = load_variable_content()
        variable = list(variable_descripttion.keys())
        num_variable = len(list(variable_descripttion.keys()))

        domain_dcb = load_domain_content(self.description)
        prompts = []

        for i in range(B):
            batch_data = x_enc[i]  # (T, N)
            batch_time = time[i]
            
            real_time_int = real_time[i].item()
            
            time_finish = str(real_time[i].item())
            time_start = str(batch_time[0].item())
            
            statistical_df = pd.DataFrame(x_enc[i][:real_time[i]].cpu())
            
            min_values_series = statistical_df.apply(min_with_na_handling, axis=0)
            max_values_series = statistical_df.apply(max_with_na_handling, axis=0)
            median_values_series = statistical_df.apply(median_with_na_handling, axis=0)
            
            min_values = min_values_series.values #(N,)
            max_values = max_values_series.values #(N,)
            median_values = median_values_series.values #(N,)
            
            batch_demo_inf = demo[i]
            age = str(batch_demo_inf[0].item())
            sex = str(batch_demo_inf[1].item())
            
            if sex == '1':
                sex = 'Male'
            else:
                sex = 'Female'
                
            if batch_demo_inf[2] or batch_demo_inf[3]:
                if batch_demo_inf[2]:
                    icu = "medical ICU"
                if batch_demo_inf[3]:
                    icu = "surgical ICU"
            
            for j in range(N):
                current_val = variable[j]
                variable_dcb = variable_descripttion[current_val]
                        
                min_values_str = min_values[j]
                max_values_str = max_values[j]
                median_values_str = median_values[j]

                last_value = 'Not measured'
                corresponding_time = 'Not measured'
                
                for i in range(len(batch_data[:real_time_int, j]) - 1, -1, -1):
                    if not torch.isnan(batch_data[:real_time_int, j][i]):
                        last_value = str(batch_data[:real_time_int, j][i].item())
                        corresponding_time = str(batch_time[:real_time_int][i].item())
                        break
                    else:
                        last_value = 'Not measured'
                        corresponding_time = 'Not measured'
                    
                prompt_ = (
                    f"<|start_prompt|>Domain description: {domain_dcb}"
                    f"Task Description: Predict whether this patient will develop sepsis in the future based on the current window information of length {str(self.seq_len)}. (The patient's actual data is up to {str(time_finish)}, and the {str(60 - int(time_finish))} hours after this should not be considered.)"
                    f"The time series information you are currently viewing is from a patient who is {sex}, {age} years old, and admitted in {icu}"
                    f"Variable description: {current_val} {variable_dcb}"
                    f"Input {current_val} statistics in current time window: "
                    f"min value {min_values_str}, "
                    f"max value {max_values_str}, "
                    f"median value {median_values_str}, "
                    f"In the current window, the last measurement value of variable {current_val} is {last_value}, and the measurement time is {corresponding_time} minutes after ICU admission.<|<end_prompt>|>"
                )
                
                prompts.append(prompt_)
                
        
        # prompts에는 B*N개의 원소가 들어가 있음
        prompt = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids # B*N, prompt_token(문장 별 최대 토큰 수)
        # self.quantized_llm.to('cpu')
        # self.quantized_llm.eval()
        with torch.no_grad():
            prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # # (B*N, prompt_token, dim(4096)
        prompt_embeddings = prompt_embeddings.reshape(B, N, prompt_embeddings.size(1), prompt_embeddings.size(2)) # (B, N, prompt_token, dim)
        
        #dimensionality rediction
        prompt_embeddings_unfolded = prompt_embeddings.unfold(dimension=2, size=10, step=4)
        prompt_embeddings_reduced = prompt_embeddings_unfolded.mean(dim=4) # B, N, 56, 4096

        words = self.vocab.split(", ")

        def get_word_embedding(word):
            inputs = self.tokenizer(word, return_tensors="pt", add_special_tokens=False)
            input_ids = inputs['input_ids']
            embeddings = self.embedding_layers(torch.tensor(input_ids).to(x_enc.device))
            
            if embeddings.shape[1] > 1:
                word_embedding = embeddings.mean(dim=1)
            else:
                word_embedding = embeddings
            
            return word_embedding.squeeze()

        word_embeddings = [get_word_embedding(word) for word in words]
        source_embeddings = torch.stack(word_embeddings).to(x_enc.device) #(vocab, 4096)
        del word_embeddings
        tkn = load_missing_token('Missing_tokens')
        tokens = tkn.split(", ")

        def get_missing_tokens(tokens):
            inputs = self.tokenizer(tokens, return_tensors="pt", add_special_tokens=False)
            input_ids = inputs['input_ids']
            embeddings =  self.embedding_layers(torch.tensor(input_ids).to(x_enc.device))
            
            if embeddings.shape[1] > 1:
                word_embedding = embeddings.mean(dim=1).mean(dim=1)
            else:
                word_embedding = embeddings
            
            return word_embedding.squeeze()

        Missing_Tokens = [get_missing_tokens(tk) for tk in tokens]
        MS_Tokens = torch.stack(Missing_Tokens).to(x_enc.device, dtype=torch.float32) #(34,)
        
        del Missing_Tokens
        nan_mask = torch.isnan(x_enc)
        for var_idx in range(x_enc.size(2)):  # 34
            x_enc[:, :, var_idx][nan_mask[:, :, var_idx]] = MS_Tokens[var_idx]
            
        x_enc = x_enc.permute(0, 2, 1) # B, T, N -> B, N, T
            
        paddings = self.tokenizer('padding', return_tensors="pt", add_special_tokens=False)
        padding_ids = paddings['input_ids']
        padding_value = self.embedding_layers(torch.tensor(padding_ids).to(x_enc.device)).mean(dim=1).mean(dim=1)

        for i in range(B):
            x_enc[i, :, real_time[i]:] = padding_value
              
        # # 평균과 표준편차 계산 (특징 차원 N에 대해 계산)
        # mean = x_enc.mean(dim=2, keepdim=True)
        # std = x_enc.std(dim=2, keepdim=True)

        # # 노말 정규화
        # x_enc = (x_enc - mean) / std
        
        enc_out = self.reprogramming_layer(x_enc, source_embeddings, source_embeddings) # B, T, N, 4096
        del source_embeddings
        llama_enc_out = torch.cat([prompt_embeddings_reduced.to(enc_out.device), enc_out.reshape(B, N, T, -1)], dim=2) # B, N, T+prompt_token, 4096
        del enc_out
        # lets_getit = llama_enc_out.reshape(B, -1, llama_enc_out.shape[-1]) # variable mixing B, N * (T+prompt_token), 4096
        # with torch.no_grad():
        #     dec_out = self.llm_model(inputs_embeds=lets_getit).hidden_states[-1] # B,  N * (T+prompt_token), 4096
        # dec_out = dec_out[:, :, :self.d_ff] # # B,  N * (T+prompt_token), d_ff
        # del lets_getit
        
    
        # dec_out = []

        # for b in range(B):
        #     # (N, T+P, 4096)
        #     for n in range(N):
        #         input_tensor = llama_enc_out[b, n, :, :].unsqueeze(0) #(N, T+P, 4096)
        #         with torch.no_grad():
        #             out = self.llm_model(inputs_embeds=input_tensor).hidden_states[-1]
        #             dec_out.append(out.squeeze(0).squeeze(0))  

        # dec_out = torch.stack(dec_out, dim=0) 
        # dec_out.reshape(B, N, dec_out.shape[-2], -1) # (B, N, T+P, 4096)
        # dec_out = dec_out.reshape(B, -1, llama_enc_out.shape[-1])
        # dec_out = dec_out[:, :, :self.d_ff]
        
        B, N, T_plus_P, embedding_dim = llama_enc_out.shape

        # llama_enc_out의 shape를 (B*N, T+P, 4096)으로 변경하여 배치로 처리
        input_tensor = llama_enc_out.view(B * N, T_plus_P, embedding_dim)

        with torch.no_grad():
            out = self.llm_model(inputs_embeds=input_tensor).hidden_states[-1]

        # -> (B, N, T+P, 4096)
        dec_out = out.view(B, N, T_plus_P, embedding_dim)
        dec_out = dec_out.reshape(B, -1, llama_enc_out.shape[-1])
        dec_out = dec_out[:, :, :self.d_ff]
        
        head_nf = self.d_ff * (self.seq_len + prompt_embeddings_reduced.shape[-2])
        
        self.output_projection = ClassificationHead(self.enc_in, head_nf, self.pred_len,
                                                 head_dropout=0.1).to(x_enc.device)
        dec_out = self.output_projection(dec_out.to(x_enc.device)) # B, pred_window

        return dec_out

class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_llm = d_llm or d_model  # Ensure d_llm has a default value

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, N, L = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        # Flatten B and N dimensions for processing
        target_embedding = target_embedding.reshape(B * N, L)
        
        # Apply projections
        target_embedding = self.query_projection(target_embedding).view(B * N, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        # Perform reprogramming
        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        # Reshape back to (B, N, L, d_llm)
        out = out.reshape(B, N, L, -1)
        # attention_scores = attention_scores.view(B, N, H, L, S)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B_N, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
    
    
