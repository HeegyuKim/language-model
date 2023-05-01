DATASET="heegyu/naver_webnovel"
TASK="gpt-finetuning"
MODEL_TYPE="causal-lm"

# MODEL_NAME="skt/kogpt2-base-v2"

train() {
    DATASET="heegyu/naver_webnovel"
    PROJECT="naver-webnovel"
    MODEL_NAME="heegyu/ajoublue-gpt2-medium"
    
    BATCH_SIZE=$1
    LR=$2

    RUN_NAME="ajoublue-gpt2-medium-$LR-$((8 * $BATCH_SIZE))"

    accelerate launch train_torch.py \
        --output_dir "./checkpoint/$TASK" \
        --project $PROJECT \
        --run_name $RUN_NAME \
        --do_eval --do_train \
        --model_name_or_path $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --task $TASK \
        --dtype bfloat16 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --learning_rate $LR \
        --gradient_accumulation_steps $BATCH_SIZE \
        --dataset_name $DATASET \
        --num_train_epochs 10 \
        --learning_rate $LR \
        --max_sequence_length 1024 \
        --save_strategy no \
        --logging_steps 50
}

train 8 1e-4
train 16 1e-4

train 8 3e-4
train 16 3e-4

train 8 5e-5
train 16 5e-5
