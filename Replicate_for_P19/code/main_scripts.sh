export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1

model_name=TimeLLM
train_epochs=300
learning_rate=0.001
llama_layers=16

master_port=0
num_process=2
batch_size=1
d_model=16
d_ff=16

comment='TimeLLM'

accelerate launch --config_file /home/DAHS2/.cache/huggingface/accelerate/default_config.yaml --num_processes 2 run_EHRTimeLLM.py \
  --is_training 1 \
  --root_path /home/DAHS2/Timellm/Replicate_for_P19/ \
  --PT_dict_path dataset/data/processed_data/PT_dict_list_6.npy \
  --outcomes_path dataset/data/processed_data/arr_outcomes_6.npy \
  --split_num 1 \
  --mode Train \
  --model_id Sepsis \
  --model $model_name \
  --data P19 \
  --lradj COS \
  --seq_len 60 \
  --pred_len 1 \
  --n_heads 1 \
  --itr 1 \
  --llm_model LLAMA \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
