# # bash ./scripts/TimeLLM_ETTh1_binary.sh 
# export TOKENIZERS_PARALLELISM=false
# export TRANSFORMERS_VERBOSITY=debug
# export ACCELERATE_LOG_LEVEL=debug
# model_name=TimeLLM-Binary
# train_epochs=2
# learning_rate=0.001
# llama_layers=32

# master_port=12346
# num_process=2
# batch_size=32
# d_model=32
# d_ff=32

# comment='TimeLLM-ETTh1_Binary'

# accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port main.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1_Binary.csv \
#   --model_id ETTh1_512_8 \
#   --model $model_name \
#   --data ETTh1_Binary \
#   --features M \
#   --seq_len 96 \
#   --label_len 8 \
#   --pred_len 8 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size $batch_size \
#   --learning_rate $learning_rate \
#   --llm_layers $llama_layers \
#   --train_epochs $train_epochs \
#   --model_comment $comment \
#   --deepspeed_config /home/DAHS2/Timellm/Replicate/ds_config_zero2.json

# accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port main.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1_Binary.csv \
#   --model_id ETTh1_512_8 \
#   --model $model_name \
#   --data ETTh1_Binary\
#   --features M \
#   --seq_len 384 \
#   --label_len 8 \
#   --pred_len 8 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des 'Exp' \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size $batch_size \
#   --learning_rate $learning_rate \
#   --llm_layers $llama_layers \
#   --train_epochs $train_epochs \
#   --model_comment $comment

# # bash ./scripts/TimeLLM_ETTh1_binary.sh 

# export TOKENIZERS_PARALLELISM=false

# model_name="TimeLLM-Binary"
# train_epochs=2
# learning_rate=0.001
# llama_layers=32

# master_port=33333
# num_process=2
# batch_size=16
# d_model=32
# d_ff=32

# comment="TimeLLM-ETTh1_Binary"

# # Ensure variables are correctly assigned
# echo "model_name: $model_name"
# echo "train_epochs: $train_epochs"
# echo "learning_rate: $learning_rate"
# echo "llama_layers: $llama_layers"
# echo "master_port: $master_port"
# echo "num_process: $num_process"
# echo "batch_size: $batch_size"
# echo "d_model: $d_model"
# echo "d_ff: $d_ff"
# echo "comment: $comment"

# # accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port main.py \
# python main.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTh1_Binary.csv \
#   --model_id ETTh1_512_8 \
#   --model $model_name \
#   --data ETTh1_Binary \
#   --features M \
#   --seq_len 384 \
#   --label_len 8 \
#   --pred_len 8 \
#   --factor 3 \
#   --enc_in 7 \
#   --dec_in 7 \
#   --c_out 7 \
#   --des "Exp" \
#   --itr 1 \
#   --d_model $d_model \
#   --d_ff $d_ff \
#   --batch_size $batch_size \
#   --learning_rate $learning_rate \
#   --llm_layers $llama_layers \
#   --train_epochs $train_epochs \
#   --model_comment "$comment"

# export TRANSFORMERS_VERBOSITY=debug
# export ACCELERATE_LOG_LEVEL=debug
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1

model_name=TimeLLM-Shock
train_epochs=300
learning_rate=0.0001
llama_layers=16

master_port=0
num_process=2
batch_size=16
d_model=16
d_ff=16

comment='TimeLLM-Shock'

accelerate launch --config_file /home/DAHS2/.cache/huggingface/accelerate/default_config.yaml --num_processes 2 /home/DAHS2/Timellm/Replicate/main.py \
  --task_name classification \
  --is_training 1 \
  --root_path /home/DAHS2/Timellm/Replicate/dataset/data \
  --data_path HiRID_shock_10min.csv.gz \
  --trn_split_path HiRID_shock_10min_trn_toy.csv.gz \
  --vld_split_path HiRID_shock_10min_vld_toy.csv.gz \
  --tst_split_path HiRID_shock_10min_tst_toy.csv.gz \
  --mode Valid \
  --model_id shock_10 \
  --model $model_name \
  --data hirid \
  --lradj COS \
  --seq_len 10 \
  --label_len 1 \
  --pred_len 1 \
  --enc_in 10 \
  --n_heads 1 \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
