export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

model_name=EHRTimeLLM
train_epochs=20
learning_rate=0.001
llama_layers=16

master_port=1041
num_process=2
batch_size=4
d_model=16
d_ff=32

comment='EHRTimeLLM'

accelerate launch --config_file /home/DAHS2/.cache/huggingface/accelerate/default_config.yaml --num_processes 2 run_EHRTimeLLM_IME.py \
  --is_training 1 \
  --root_path /home/DAHS2/Timellm/Replicate_for_P19/ \
  --PT_dict_path dataset/data/processed_data/PT_dict_list_6.npy \
  --outcomes_path dataset/data/processed_data/arr_outcomes_6.npy \
  --split_num 4 \
  --mode Train \
  --model_id Sepsis \
  --model $model_name \
  --data P19 \
  --lradj PEMS \
  --seq_len 60 \
  --pred_len 1 \
  --n_heads 8 \
  --itr 5 \
  --llm_model GPT2 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment 