export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1

model_name=EHRTimeLLM_Reprogramming_visualization

llama_layers=16

master_port=0
num_process=2
batch_size=48
d_model=16
d_ff=16

comment='EHRTimeLLM'

accelerate launch --config_file /home/DAHS2/.cache/huggingface/accelerate/default_config.yaml --num_processes 2 inf_EHRTimeLLM_v3.py \
  --is_training 1 \
  --root_path /home/DAHS2/Timellm/Replicate_for_P19/ \
  --PT_dict_path dataset/data/processed_data/PT_dict_list_6.npy \
  --outcomes_path dataset/data/processed_data/arr_outcomes_6.npy \
  --trained_model_path code/checkpoints/Sepsis_EHRTimeLLM_P19_60_pl1_dm16_nh1_df16_il_0-EHRTimeLLM/checkpoint_split_1 \
  --split_num 1 \
  --model_id Sepsis \
  --model $model_name \
  --data P19 \
  --mode Visualization \
  --seq_len 60 \
  --pred_len 1 \
  --n_heads 1 \
  --llm_model GPT2 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --llm_layers $llama_layers \
  --model_comment $comment 