export WANDB_PROJECT="gpt2-dialog"

DTYPE="float32"

function train {
    MODEL_NAME=$1
    RUN_NAME="$MODEL_NAME"
    python3 train_clm_flax_v2.py \
        --run_name=$RUN_NAME \
        --output_dir="checkpoint/$RUN_NAME" \
        --model_name=$MODEL_NAME \
        --train_file="/data/dialog-v1-vocab51k-block1024/train/*.jsonl" \
        --validation_file="/data/dialog-v1-vocab51k-block1024/test/*.jsonl" \
        --cache_dir="/data/.cache/" \
        --do_train \
        --do_eval \
        --dtype=$DTYPE \
        --block_size="1024" \
        --per_device_train_batch_size="1" \
        --per_device_eval_batch_size="1" \
        --gradient_accumulation_steps=8 \
        --learning_rate="1e-4" \
        --overwrite_output_dir \
        --num_train_epochs="3" \
        --logging_steps="100" \
        --save_strategy="no" \
        --eval_strategy="epoch"
}

train "heegyu/kogpt-j-base"