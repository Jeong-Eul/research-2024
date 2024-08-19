export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1

model_name=EHRTimeLLM
llama_layers=16

master_port=0
num_process=2
batch_size=2
d_model=16
d_ff=16

comment='EHRTimeLLM'

accelerate launch --config_file /home/DAHS2/.cache/huggingface/accelerate/default_config.yaml --num_processes 2 inf_EHRTimeLLM.py \
  --is_training 1 \
  --root_path /home/DAHS2/Timellm/Replicate_for_P19/ \
  --PT_dict_path dataset/data/processed_data/PT_dict_list_6.npy \
  --outcomes_path dataset/data/processed_data/arr_outcomes_6.npy \
  --trained_model_path code/model_checkpoint/Best_model_epoch_1_loss-0.15875975988882104.pt \
  --split_num 1 \
  --model_id Sepsis \
  --model $model_name \
  --data P19 \
  --seq_len 60 \
  --pred_len 1 \
  --n_heads 1 \
  --llm_model GPT2 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --llm_layers $llama_layers \
  --model_comment $comment 