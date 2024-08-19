from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, AutoTokenizer, AutoModelForCausalLM
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content, load_vocabulary, load_domain_content, load_variable_content, load_missing_token

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
        # nf = d_ff * patch num
        self.window = target_window
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear_1 = nn.Linear(nf * n_vars , nf *n_vars)
        self.linear_2 = nn.Linear(nf *n_vars, target_window)
        
        self.batch_norm1 = nn.BatchNorm1d(nf * n_vars)
        self.batch_norm2 = nn.BatchNorm1d(target_window)
        
        self.dropout = nn.Dropout(head_dropout)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)

    def forward(self, x):
        B, _, _ = x.shape
        x = self.flatten(x) # B, N * dim * patch num
        
        # residual = x
        x = self.linear_1(x)
        x = self.dropout(x)
        x = self.batch_norm1(x)
        # x = self.relu(x)
        # x = x + residual
        
        x = self.linear_2(x)
        # x = self.batch_norm2(x)
        # x = self.dropout(x)
        # x = self.sigmoid(x)
        # x = x.view(B, self.window, 1)
        
        return x



class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.vocab = configs.vocab

        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained("huggyllama/llama-7b")
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    "huggyllama/llama-7b",
                    trust_remote_code=True,
                    # local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True,
                )
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
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    # local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
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
            self.description = configs.domain
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.embedding_layers = self.llm_model.get_input_embeddings()
        # self.vocab_size = self.word_embeddings.shape[0]
        # self.num_tokens = 5
        # self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.seq_len, configs.n_heads, None, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        # if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
        #     self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
        #                                          head_dropout=configs.dropout)
        # elif self.task_name == 'classification':
        #     self.output_projection = ClassificationHead(configs.enc_in, self.head_nf, self.pred_len,
        #                                          head_dropout=configs.dropout)
        # else:
        #     raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)
        self.enc_in = configs.enc_in
    

    def forward(self, x_enc, time, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # x_enc = x_enc.to(dtype=torch.bfloat16)
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]
        elif self.task_name == 'classification':
            # x_enc = x_enc.to(dtype=torch.bfloat16)
            dec_out = self.forecast(x_enc, time)
            return dec_out #B, pred_window
        return None

    def forecast(self, x_enc, time):

        x_enc = x_enc.to(dtype=torch.float32)
        time = time.to(dtype=torch.float32)
        B, T, N = x_enc.size()  # (B, 10, 22)

        # variable = ['ABPd', 'ABPs', 'Creatinine', 'HR', 'INR', 'Lactate', 'MAP',
        #     'PaO2', 'Platelet_Count', 'Respiratory_rate', 'SpO2', 'pH',
        #     'Time_since_ICU_admission', 'Height', 'Weight', 'Bilirubin',
        #     'FiO2', 'Temperature', 'Troponin-T', 'Cardiac output', 'Sex', 'Age']
        
        variable = ['HR', 'Lactate', 'MAP', 'Respiratory_rate', 'SpO2', 'Time_since_ICU_admission', 'Height', 'Weight', 'Sex', 'Age']
        
        variable_pass = ['Time_since_ICU_admission', 'Height', 'Weight', 'Sex', 'Age']

        
        num_variable = len(variable)
        prompts = []

        for i in range(B):
            batch_data = x_enc[i]  # (T, N)
            batch_time = time[i]
            # 각 변수에 대해 통계값 계산
            # min_values = torch.min(batch_data, dim=0)[0]
            # max_values = torch.max(batch_data, dim=0)[0]
            # medians = torch.median(batch_data, dim=0).values
            # trends = batch_data.diff(dim=0).sum(dim=0)
            time_start = str(batch_time[0].item())
            time_finish = str(batch_time[-1].item())
            
            for j in range(N):
                current_val = variable[j]
                
                if current_val in variable_pass:
                    continue
                
                else:
                
                    age = str(torch.unique(batch_data[:, -1], dim=0).item())
                    sex = str(torch.unique(batch_data[:, -2], dim=0).item())
                    
                    if torch.unique(batch_data[:, -4], dim=0).numel() == 1:
                        # 고유 값이 하나만 있는 경우
                        height_str = str(torch.unique(batch_data[:, -4], dim=0).item())
                    else:
                        # 고유 값이 여러 개인 경우, 값을 문자열로 변환
                        height_str = ", ".join(map(str, torch.unique(batch_data[:, -4], dim=0).tolist()))
             
                    if torch.unique(torch.unique(batch_data[:, -3], dim=0)).numel() == 1:
                        # 고유 값이 하나만 있는 경우
                        weight_str = str(torch.unique(torch.unique(batch_data[:, -3], dim=0)).item())
                    else:
                        # 고유 값이 여러 개인 경우, 값을 문자열로 변환
                        weight_str = ", ".join(map(str, torch.unique(torch.unique(batch_data[:, -3], dim=0)).tolist()))
                    
                    if sex == '1':
                        sex = 'Male'
                    else:
                        sex = 'Female'
                    variable_dcb = load_variable_content(current_val)
                        
                    # min_values_str = str(min_values[j].item())
                    
                    # if min_values_str == 'nan':
                    #     min_values_str = 'Not measured'
                    # max_values_str = str(max_values[j].item())
                    
                    # if max_values_str == 'nan':
                    #     max_values_str = 'Not measured'
                    # median_values_str = str(medians[j].item())
                    
                    # if median_values_str == 'nan':
                    #     median_values_str = 'Not measured'
                        
                    # trend_str = 'upward' if trends[j] > 0 else 'downward'
                    # if trend_str == 'nan':
                    #     trend_str = 'Not measured'
                    
                    last_value = 'Not measured'
                    corresponding_time = 'Not measured'
                    
                    for i in range(len(batch_data[:, j]) - 1, -1, -1):
                        if not torch.isnan(batch_data[:, j][i]):
                            last_value = str(batch_data[:, j][i].item())
                            corresponding_time = str(batch_time[i].item())
                            break
                        else:
                            last_value = 'Not measured'
                            corresponding_time = 'Not measured'
                    
                    prompt_ = (
                        f"Domain description: {self.description}"
                        f"Task description: Predict whether a shock will occur within the next 6 hours based on the current window information of length {str(self.seq_len)}. "
                        f"The time series information(Time window unit start point is {time_start} minutes from ICU admission and end point is {time_finish} minutes) you are currently viewing is from a patient who is {sex}, {age} years old. This patient's height is {height_str} cm and weight is {weight_str} kg. "
                        f"Variable description: {variable_dcb}"
                        f"Input {current_val} statistics in current time window: "
                        f"In the current window, the last measurement value of variable {current_val} is {last_value}, and the measurement time is {corresponding_time} minutes after ICU admission.<|<end_prompt>|> "
                    )
                    
                    # f"min value of {current_val} in the batch sequence is {min_values_str}, "
                    #     f"max value of {current_val} in the batch sequence is {max_values_str}, "
                    #     f"median value of {current_val} in the batch sequence is {median_values_str}, "
                    #     f"the trend of {current_val} in the batch sequence is {trend_str}, "
                    
                    prompts.append(prompt_)
                
        x_enc = self.normalize_layers(x_enc, 'norm')
        
        # prompts에는 B*N개의 원소가 들어가 있음
        prompt = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids # B*(N-variable pass), prompt_token(문장 별 최대 토큰 수)
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # # (B*N, prompt_token, dim(4096)
        prompt_embeddings = prompt_embeddings.view(B, N-len(variable_pass), prompt_embeddings.size(1), prompt_embeddings.size(2)) # (B, N-len(variable_pass), prompt_token, dim)
        
        #dimensionality rediction
        prompt_embeddings_unfolded = prompt_embeddings.unfold(dimension=2, size=20, step=4)
        prompt_embeddings_reduced = prompt_embeddings_unfolded.mean(dim=4) # B, N-variable pass, 56, 4096
        
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
        
        tkn = load_missing_token('Missing_tokens_reduced')
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
        MS_Tokens = torch.stack(Missing_Tokens).to(x_enc.device, dtype=torch.float32) #(22,)
        
        nan_mask = torch.isnan(x_enc)

        # 결측값 대체
        for var_idx in range(x_enc.size(1)):  # 변수 갯수 (22개)
            x_enc[:, :, var_idx][nan_mask[:, :, var_idx]] = MS_Tokens[var_idx]

        
        # Exclude variable pass before patch embedding 
        exclude_indices = [5, 6, 7, 8, 9]

        include_indices = [i for i in range(x_enc.size(2)) if i not in exclude_indices]
        filtered_tensor = x_enc[:, :, include_indices]
        filtered_tensor = filtered_tensor.permute(0, 2, 1).contiguous().to(dtype=torch.bfloat16)
        
        # enc_out, n_vars = self.patch_embedding(filtered_tensor) # B, (N-variable pass), patch_num, patch_len : patch 하나가 patch len 만큼 임베딩 됨
        enc_out = self.reprogramming_layer(filtered_tensor, source_embeddings, source_embeddings) # B, (N-variable pass), patch_num -> seq len, 4096
        # enc_out = enc_out.to(dtype = torch.float32)
        # prompt_embeddings = prompt_embeddings.to(dtype = torch.float32)
        llama_enc_out = torch.cat([prompt_embeddings_reduced, enc_out], dim=2) # B, (N-variable pass), patch_num+prompt_token, 4096
        
        lets_getit = llama_enc_out.reshape(B, -1, llama_enc_out.shape[-1]) # variable mixing B, (N-variable pass) * (patch_num+prompt_token), 4096
        
        dec_out = self.llm_model(inputs_embeds=lets_getit).hidden_states[-1] # B,  N * (patch_num+prompt_token), 4096
        dec_out = dec_out[:, :, :self.d_ff] # # B,  N * (patch_num+prompt_token), d_ff
        del prompt, prompt_, enc_out, source_embeddings
        
        head_nf = self.d_ff * (self.seq_len + prompt_embeddings_reduced.shape[-2])
        
        self.output_projection = ClassificationHead(self.enc_in-len(variable_pass), head_nf, self.pred_len,
                                                 head_dropout=0.1).to(filtered_tensor.device)
        
        dec_out = self.output_projection(dec_out) # B, pred_window

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
        target_embedding = target_embedding.view(B * N, L)
        
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