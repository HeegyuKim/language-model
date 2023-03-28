DATASET="heegyu/naver_webnovel"
TASK="gpt-finetuning"
MODEL_TYPE="causal-lm"
# MODEL_NAME="heegyu/ajoublue-gpt2-medium"
MODEL_NAME="skt/kogpt2-base-v2"

train() {
    DATASET="heegyu/naver_webnovel" # "imdb"
    RUN_NAME="$MODEL_NAME-$DATASET"
    PROJECT="naver-webnovel"
    BATCH_SIZE=$1

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
        --gradient_accumulation_steps $BATCH_SIZE \
        --dataset_name $DATASET \
        --num_train_epochs 10 \
        --max_sequence_length 1024 \
        --save_strategy epoch \
        --logging_steps 50
}

train 8
