

export WANDB_PROJECT="gpt2"

RUN_NAME="gpt-j-base-v1-24L-lr6e-4-batch4-rev1"


python3 train_clm_flax_v2.py \
    --run_name=$RUN_NAME \
    --output_dir="checkpoint/$RUN_NAME" \
    --tokenizer_name="heegyu/kogpt-j-base-24L" \
    --config_name="heegyu/kogpt-j-base-24L" \
    --train_file="/data/gpt-data/v1-vocab51k-block1024/*.jsonl" \
    --cache_dir="/data/gpt-data/.cache/" \
    --do_train \
    --block_size="1024" \
    --per_device_train_batch_size="4" \
    --per_device_eval_batch_size="4" \
    --learning_rate="6e-4" \
    --warmup_steps="1000" \
    --adam_beta1="0.9" --adam_beta2="0.98" --weight_decay="0.01" \
    --overwrite_output_dir \
    --num_train_epochs="3" \
    --logging_steps="500" \
    --save_steps="25000"