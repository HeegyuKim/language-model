

export WANDB_PROJECT="gpt2"

RUN_NAME="gpt-j-base-v0-lr6e-4-batch8"

python3 train_clm_flax_v2.py \
    --run_name=$RUN_NAME \
    --output_dir="checkpoint/$RUN_NAME" \
    --tokenizer_name="heegyu/kogpt-j-base" \
    --config_name="heegyu/kogpt-j-base" \
    --train_file="data/v0-vocab51k-block1024/*.jsonl" \
    --do_train \
    --block_size="1024" \
    --per_device_train_batch_size="8" \
    --per_device_eval_batch_size="8" \
    --learning_rate="1e-4" \
    --warmup_steps="1000" \
    --adam_beta1="0.9" --adam_beta2="0.98" --weight_decay="0.01" \
    --overwrite_output_dir \
    --num_train_epochs="3" \
    --logging_steps="1" \
    --save_steps="10000"