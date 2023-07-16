export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export USE_Torch=True


TASK="nia-dialog-v2"
MODEL_TYPE="causal-lm"
PROJECT="nia-dialog"


train() {
    MODEL_NAME=$1
    BATCH_SIZE=$2
    RUN_NAME="v2-$1-s1024-b$(($BATCH_SIZE * 8))-bf16"

    accelerate launch train_torch.py \
        --output_dir "./checkpoint/$TASK" \
        --project $PROJECT \
        --run_name "$RUN_NAME" \
        --do_eval --do_train \
        --model_name_or_path $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --task $TASK \
        --dtype bfloat16 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps $BATCH_SIZE \
        --num_train_epochs 5 \
        --max_sequence_length 1024 \
        --save_strategy epoch \
        --logging_steps 50
}

train "heegyu/ajoublue-gpt2-medium" 8
