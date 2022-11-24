export WANDB_PROJECT="gpt2"

MODEL_NAME="heegyu/kogpt-neox-small"
RUN_NAME="kogpt-neox-small"
BLOCK_SIZE=512
BATCH_SIZE=16
OVERWRITE=true

python train_clm_pt.py \
    --run_name $RUN_NAME \
    --config_name $MODEL_NAME \
    --tokenizer_name $MODEL_NAME \
    --train_data_dir "data/test/" \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 1 \
    --save_steps 50000 \
    --eval_steps 10000 \
    --max_steps 500000 \
    --fp16 true \
    --block_size $BLOCK_SIZE \
    --deepspeed deepspeed/stage2.json \
    --output_dir "checkpoint/$RUN_NAME" \
    --overwrite_output_dir $OVERWRITE