export WANDB_PROJECT="gpt2"

MODEL_NAME="heegyu/kogpt-neox-tiny"
RUN_NAME="kogpt-neox-tiny"
BLOCK_SIZE=512
BATCH_SIZE=32

# MODEL_NAME="skt/kogpt2-base-v2"
# RUN_NAME="kogpt2-base-v2"
# BLOCK_SIZE=1024
# BATCH_SIZE=2

# DATASET_NAME="datasets.json"
# DATASET_NAME="datasets_test.json"
OVERWRITE=true

# --dataset_file $DATASET_NAME \
# --do_eval \
python train_clm_pt.py \
    --config_name $MODEL_NAME \
    --tokenizer_name $MODEL_NAME \
    --train_file "data/test/*.json" \
    --shuffle_train true \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --do_train \
    --gradient_accumulation_steps 1 \
    --save_steps 10000 \
    --eval_steps 10000 \
    --fp16 true \
    --block_size $BLOCK_SIZE \
    --deepspeed deepspeed/stage2.json \
    --output_dir "/tmp/$RUN_NAME" \
    --overwrite_output_dir $OVERWRITE