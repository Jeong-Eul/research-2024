from math import sqrt

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel
import transformers
from utils.tools import load_content, load_vocabulary, load_domain_content, load_variable_content, load_missing_token, min_with_na_handling, max_with_na_handling, median_with_na_handling
from layers.Embed import PatchEmbedding, PromptPatchEmbedding

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
    def __init__(self, n_vars, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        # self.flatten = nn.Flatten(start_dim=-2)
        self.linear_1 = nn.Linear(n_vars , target_window)
        self.dropout = nn.Dropout(head_dropout)
        
        nn.init.xavier_uniform_(self.linear_1.weight)
        
    def forward(self, x):
        
        # x = self.flatten(x)
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
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.d_model = configs.d_model

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

        self.patch_num = int((self.seq_len - self.patch_len)/self.stride + 2)
        
        self.dropout = nn.Dropout(configs.dropout)
        self.embedding_layers = self.llm_model.get_input_embeddings()
        self.reprogramming_layer = ReprogrammingLayer(self.d_model, configs.n_heads, None, self.d_llm)
        self.patch_embedding = PatchEmbedding(configs.d_model, self.patch_len, self.stride, configs.dropout).float()
        
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        
        self.prompt_patching = PromptPatchEmbedding(input_dim = configs.llm_dim, output_dim = configs.llm_dim, target_len = self.patch_num).float()
        self.enc_in = configs.enc_in
        self.head_nf = self.d_ff * self.patch_num
        # self.head_nf = 5488
        self.output_projection = ClassificationHead(self.enc_in, self.pred_len,
                                                 head_dropout=0.1)
        self.d_llm = configs.llm_dim
    
    def forward(self, x_enc, time, real_time, demo, mask=None):
        dec_out = self.forecast(x_enc, time, real_time, demo)
        return dec_out #B, pred_window

    def forecast(self, x_enc, time, real_time, demo):

        x_enc = x_enc.to(dtype=torch.float32)
        time = time.to(dtype=torch.float32)
        B, T, N = x_enc.size()  # (B, 60, 34)

        vital_index = np.arange(34)[:8]
        lab_index = np.arange(34)[8:]
        
        #Prompt Embedding
        variable_descripttion = load_variable_content()
        variable = list(variable_descripttion.keys())
        
        domain_dcb = load_domain_content('Sepsis')
        prompts = []

        for i in range(B):
            batch_data = x_enc[i]  # (T, N)
            batch_time = time[i]
          
            real_time_int = int(real_time[i].item())
            statistical_df = pd.DataFrame(batch_data[:real_time_int].cpu()).replace(-0.0, np.nan)
            
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
            
            for j in lab_index:
                current_val = variable[j]
                variable_dcb = variable_descripttion[current_val]
                        
                min_values_str = min_values[j]
                max_values_str = max_values[j]
                median_values_str = median_values[j]

                last_value = 'Not measured'
                corresponding_time = 'Not measured'
                
                for i in range(len(statistical_df.iloc[:, j]) - 1, -1, -1):
                    if not np.isnan(statistical_df.iloc[:, j][i]):
                        last_value = str(statistical_df.iloc[:, j][i].item())
                        corresponding_time = str(batch_time[:real_time_int][i].item())
                        break
                    else:
                        last_value = 'Not measured'
                        corresponding_time = 'Not measured'
                    
                prompt_ = (
                    f"<|start_prompt|>Domain description: {domain_dcb}"
                    f"Task Description: Predict whether this patient will develop sepsis in the future based on the current information. (The patient's actual data is {str(real_time_int)} hours, after this should not be considered.) "
                    f"The time series information you are currently viewing is from a patient who is {sex}, {age} years old, and admitted in {icu} "
                    f"Variable description: {current_val} {variable_dcb} "
                    f"Input {current_val} statistics in current time window: "
                    f"min value {min_values_str}, "
                    f"max value {max_values_str}, "
                    f"median value {median_values_str}, "
                    f"In the current window, the last measurement value of variable {current_val} is {last_value}, and the measurement time is {corresponding_time} minutes after ICU admission.<|<end_prompt>|>"
                )
                
                prompts.append(prompt_)
                
        prompt = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids # B*N, prompt_token(문장 별 최대 토큰 수)
        
        with torch.no_grad():
            prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # # (B*N, prompt_token, d_llm)
        prompt_embeddings = self.prompt_patching(prompt_embeddings.to(x_enc.device)) # (B_N, patch_num, d_llm)
        
        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_vital = x_enc[:, :, vital_index]
        x_vital = x_vital.permute(0, 2, 1) # B, T, N -> B, N, T
        
        #Time series patch embedding and Masking
        enc_out, n_vars = self.patch_embedding(x_vital) # B_N_v, patch_num, d_model
        
        lengths = torch.sum(time > 0, dim=1).cpu()[:, None].squeeze(-1) #B_N_v, 1
        batch_size, num_patches, _ = enc_out.shape
        replicated_lengths = lengths.repeat_interleave(len(vital_index), dim=0)
        mask = torch.arange(num_patches).unsqueeze(0).expand(batch_size, -1) >= ((replicated_lengths - self.patch_len) / self.stride + 2) #B_N_v, patch num
        enc_out = enc_out.masked_fill(mask.unsqueeze(-1).to(x_enc.device), 0)    
        
        #Reprogramming
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings, mask) # B_N_v, patch_num, d_llm
        
        #LLM Body
        llama_enc_out = torch.cat([prompt_embeddings.reshape(B, len(lab_index), self.patch_num, -1), enc_out.reshape(B, len(vital_index), self.patch_num, -1)], dim=1) # B_N, patch_num, d_llm
        input_tensor = llama_enc_out.reshape(B*N, self.patch_num, -1)
        
        with torch.no_grad():
            out = self.llm_model(inputs_embeds=input_tensor).hidden_states[-1] # B_N, patch_num, d_llm
            
        out = torch.reshape(
            out, (-1, N, out.shape[-2], out.shape[-1])) # B, N, patch_num, d_llm

        #Average pooling for sequence
        lengths2 = lengths.unsqueeze(1).to(x_enc.device) # B, 1, 1
        out = torch.sum(out, dim=2) / (lengths2 + 1) # B, N, d_llm

        # Average pooling for d_llm
        out = torch.mean(out, dim=-1) # B, N
        
        #Classifier
        dec_out = self.output_projection(out.to(x_enc.device)) # B, pred_window

        return dec_out

class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads, bias=False) # mask 된 부분은 아예 무시 되도록 bias 계산 x
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm, bias = False) # mask 된 부분은 아예 무시 되도록 bias 계산 x
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding, mask):
        B_N, L, _ = target_embedding.shape # B_N, patch_num, d_model
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B_N, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding, mask)

        out = out.reshape(B_N, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding, mask):
        B_N, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        
        expanded_mask = mask.unsqueeze(1).unsqueeze(-1).float().to(scores.device)
        
        # Mask the softmax output
        A = A * (1-expanded_mask)

        # Normalize again to ensure the masked values don't affect the sum
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-9)
        
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
    
    
