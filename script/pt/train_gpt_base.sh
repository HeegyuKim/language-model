
export WANDB_PROJECT="gpt2"
RUN_NAME="gpt-j-base-test"


# /data/v1-vocab51k-block1024/heegyu__kowikitext.jsonl
# /data/v1-vocab51k-block1024/*.jsonl

python3 train_clm_hftrainer.py \
    --run_name=$RUN_NAME \
    --output_dir="checkpoint/$RUN_NAME" \
    --tokenizer_name="heegyu/kogpt-j-base" \
    --config_name="heegyu/kogpt-j-base" \
    --train_file="/data2/v1-vocab51k-block1024/heegyu__kowikitext.jsonl" \
    --cache_dir="/data/.cache/" \
    --do_train \
    --block_size="1024" \
    --per_device_train_batch_size="1" \
    --per_device_eval_batch_size="1" \
    --gradient_accumulation_steps=64 \
    --dtype=bfloat16 \
    --learning_rate="6e-4" \
    --warmup_steps="1000" \
    --adam_beta1="0.9" --adam_beta2="0.98" --weight_decay="0.01" \
    --overwrite_output_dir \
    --num_train_epochs="3" \
    --logging_steps="2500" \
    --save_steps="500000" \
    --report_to=none