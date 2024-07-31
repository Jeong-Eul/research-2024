from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, AutoTokenizer, AutoModelForCausalLM
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

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
        # x = self.linear_1(x)
        
        # x = self.batch_norm1(x)
        # x = self.relu(x)
        # x = x + residual
        
        x = self.linear_2(x)

        # x = self.batch_norm2(x)
        
        x = self.dropout(x)
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
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.embedding_layers = self.llm_model.get_input_embeddings()
        # self.vocab_size = self.word_embeddings.shape[0]
        # self.num_tokens = 5
        # self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

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

    def forward(self, x_enc, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # x_enc = x_enc.to(dtype=torch.bfloat16)
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]
        elif self.task_name == 'classification':
            # x_enc = x_enc.to(dtype=torch.bfloat16)
            dec_out = self.forecast(x_enc)
            return dec_out #B, pred_window
        return None

    def forecast(self, x_enc):

        # # x_enc = self.normalize_layers(x_enc, 'norm')
        # x_enc = x_enc.to(dtype=torch.float32)
        # B, T, N = x_enc.size() #(B, 128, 6)
        # # x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
        # x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(N, B * T)

        # min_values = torch.min(x_enc, dim=1)[0] # 변수 별로, T만큼의 시퀀스가 존재하는 torch array임. dim = 1 방향은 각 변수 별 시퀀스를 의미, 따라서 각 변수별로 통계값이 산출됨
        # max_values = torch.max(x_enc, dim=1)[0]
        # medians = torch.median(x_enc, dim=1).values
        # # lags = self.calcute_lags(x_enc)
        # trends = x_enc.diff(dim=1).sum(dim=1)
        # # The above code is not doing anything. It contains a comment "# Python" followed by a line
        # # with the word "variable" and then more comment symbols "
        # variable = ['High UseFul Load', 'High UseLess Load', 'Medium UseFul Load', 'Medium UseLess Load', 'Low UseFul Load', 'Low UseLess Load']
        # num_variable = len(variable)
        # prompt = []
        
        # for b in range(x_enc.shape[0]):
        #     current_val = variable[b]
        #     min_values_str = str(min_values.tolist()[0])
        #     max_values_str = str(max_values.tolist()[0])
        #     median_values_str = str(medians.tolist()[0])
        #     # lags_values_str = str(lags[b].tolist())
        #     # prompt_ = (
        #     #     f"<|start_prompt|>Dataset description: {self.description}"
        #     #     f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
        #     #     "Input statistics: "
        #     #     f"min value {min_values_str}, "
        #     #     f"max value {max_values_str}, "
        #     #     f"median value {median_values_str}, "
        #     #     f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
        #     #     f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
        #     # )
        #     prompt_ = (
        #         f"<|start_prompt|>Dataset description: {self.description}"
        #         f"Task description: classify each of the next {str(self.pred_len)} based on information from the previous {str(self.seq_len)} steps using {str(num_variable)} variables in all of batch sequence; "
        #         "Input statistics: "
        #         f"min value of {current_val} in all of batch sequence is {min_values_str}, "
        #         f"max value of {current_val} in all of batch sequence is  {max_values_str}, "
        #         f"median value of {current_val} in all of batch sequence is  {median_values_str}, "
        #         f"the trend of {current_val} in all of batch sequence is {'upward' if trends[b] > 0 else 'downward'}<|<end_prompt>|> "
        #         # f"top 5 lags of {current_val} in all of batch sequence : {lags_values_str}<|<end_prompt>|>"
        #     )

        #     prompt.append(prompt_) # data description 설명은 공통으로 변수 별 차이 없이 들어감 !, 또한 prompt_는 변수의 갯수만큼 만들어짐 # N, prompt_tokens
        
        
        x_enc = x_enc.to(dtype=torch.float32)
        B, T, N = x_enc.size()  # (B, 10, 6)

        # 각 배치에 대해 개별적으로 프롬프트 생성
        variable = ['High UseFul Load', 'High UseLess Load', 'Medium UseFul Load', 'Medium UseLess Load', 'Low UseFul Load', 'Low UseLess Load']
        num_variable = len(variable)
        prompts = []
        
        for i in range(B):
            batch_data = x_enc[i]  # (T, N)
            
            # 각 변수에 대해 통계값 계산
            min_values = torch.min(batch_data, dim=0)[0]
            max_values = torch.max(batch_data, dim=0)[0]
            medians = torch.median(batch_data, dim=0).values
            trends = batch_data.diff(dim=0).sum(dim=0)
            
            for j in range(N):
                current_val = variable[j]
                min_values_str = str(min_values[j].item())
                max_values_str = str(max_values[j].item())
                median_values_str = str(medians[j].item())
                trend_str = 'upward' if trends[j] > 0 else 'downward'
                # 이렇게 해야 pad token 추가 없이 차원이 맞음
                prompt_ = (
                    f"Dataset description: {self.description} "
                    f"Task description: classify each of the next {str(self.pred_len)} based on information from the previous {str(self.seq_len)} steps; "
                    "Input statistics: "
                    f"min value of {current_val} in the batch sequence is {min_values_str}, "
                    f"max value of {current_val} in the batch sequence is {max_values_str}, "
                    f"median value of {current_val} in the batch sequence is {median_values_str}, "
                    f"the trend of {current_val} in the batch sequence is {trend_str}<|<end_prompt>|> "
                )
                
                prompts.append(prompt_)
                
        x_enc = self.normalize_layers(x_enc, 'norm')
        
        # prompts에는 B*N개의 원소가 들어가 있음
        prompt = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids # B*N, prompt_token(문장 별 최대 토큰 수)
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # # (B*N, prompt_token, dim(4096)
        prompt_embeddings = prompt_embeddings.view(B, N, prompt_embeddings.size(1), prompt_embeddings.size(2)) # (B, N, prompt_token, dim)
        
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

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc) # B, N, patch_num, patch_len : patch 하나가 patch len 만큼 임베딩 됨
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings) # B, N, patch_num, 4096
        # enc_out = enc_out.to(dtype = torch.float32)
        # prompt_embeddings = prompt_embeddings.to(dtype = torch.float32)
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=2) # B, N, patch_num+prompt_token, 4096
        
        llama_enc_out.reshape(B, -1, llama_enc_out.shape[-1]).shape # variable mixing B, N * (patch_num+prompt_token), 4096
        
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).hidden_states[-1] # B,  N * (patch_num+prompt_token), 4096
        dec_out = dec_out[:, :, :self.d_ff] # # B,  N * (patch_num+prompt_token), d_ff
        del prompt, prompt_, enc_out, source_embeddings
        
        head_nf = args.d_ff * (self.patch_nums + prompt_embeddings.shape[-2])
        
        self.output_projection = ClassificationHead(configs.enc_in, head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        
        dec_out = self.output_projection(dec_out) # B, pred_window

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


# class ReprogrammingLayer(nn.Module):
#     def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
#         super(ReprogrammingLayer, self).__init__()

#         d_keys = d_keys or (d_model // n_heads)

#         self.query_projection = nn.Linear(d_model, d_keys * n_heads)
#         self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
#         self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
#         self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
#         self.n_heads = n_heads
#         self.dropout = nn.Dropout(attention_dropout)

#     def forward(self, target_embedding, source_embedding, value_embedding):
#         B, L, _ = target_embedding.shape
#         S, _ = source_embedding.shape
#         H = self.n_heads

#         target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
#         source_embedding = self.key_projection(source_embedding).view(S, H, -1)
#         value_embedding = self.value_projection(value_embedding).view(S, H, -1)

#         out = self.reprogramming(target_embedding, source_embedding, value_embedding)

#         out = out.reshape(B, L, -1)

#         return self.out_projection(out) # (B, N, T, 4096)

#     def reprogramming(self, target_embedding, source_embedding, value_embedding):
#         B, L, H, E = target_embedding.shape

#         scale = 1. / sqrt(E)

#         scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

#         A = self.dropout(torch.softmax(scale * scores, dim=-1))
#         reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

#         return reprogramming_embedding

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
        B, N, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        # Flatten B and N dimensions for processing
        target_embedding = target_embedding.view(B * N, L, -1)
        
        # Apply projections
        target_embedding = self.query_projection(target_embedding).view(B * N, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        # Perform reprogramming
        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        # Reshape back to (B, N, L, d_llm)
        out = out.view(B, N, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B_N, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding