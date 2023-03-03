# cd data/
# python3 generate_for_gpt_dialog.py
# cd ..

export WANDB_PROJECT="gpt2-dialog"

DTYPE="float32"
REVISION="main"

function train {
    MODEL_NAME=$1
    RUN_NAME="$MODEL_NAME"
    python3 train_clm_flax_v2.py \
        --run_name=$RUN_NAME \
        --output_dir="/data/checkpoint/dialog/$RUN_NAME" \
        --model_name=$MODEL_NAME \
        --train_file="/data/dialog-v1-vocab51k-block1024/train/*.jsonl" \
        --validation_file="/data/dialog-v1-vocab51k-block1024/test/*.jsonl" \
        --cache_dir="/data/.cache/" \
        --do_train \
        --do_eval \
        --revision $REVISION \
        --dtype=$DTYPE \
        --block_size="1024" \
        --per_device_train_batch_size="1" \
        --per_device_eval_batch_size="1" \
        --gradient_accumulation_steps=8 \
        --learning_rate="1e-4" \
        --overwrite_output_dir \
        --num_train_epochs="3" \
        --logging_steps="500" \
        --save_strategy="last" \
        --eval_strategy="epoch"
}

# REVISION="master"
# train "heegyu/kogpt-j-base"
train_nsmc "skt/kogpt2-base-v2"
# train "heegyu/kogpt-j-base-24L"
# train "heegyu/ajoublue-gpt2-base"
# train "heegyu/ajoublue-gpt2-base-24L"

# DTYPE="bfloat16"
# train "heegyu/kogpt-j-350m"
# train "heegyu/ajoublue-gpt2-medium"