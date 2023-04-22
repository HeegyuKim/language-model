TASK="gorani"
MODEL_TYPE="causal-lm"
PROJECT="gorani"


train() {
    MODEL_NAME=$1
    RUN_NAME=$1
    BATCH_SIZE=$2

    accelerate launch train_torch.py \
        --output_dir "./checkpoint/$TASK" \
        --project $PROJECT \
        --run_name $RUN_NAME \
        --do_eval --do_train \
        --model_name_or_path $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --task $TASK \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps $BATCH_SIZE \
        --num_train_epochs 10 \
        --max_sequence_length 1024 \
        --save_strategy epoch \
        --logging_steps 50
}

train "heegyu/ajoublue-gpt2-medium" 8
