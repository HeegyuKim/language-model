# DATASET="heegyu/news-category-balanced-top10"
MODEL_TYPE="causal-lm"
MODEL_NAME="gpt2"

train() {
    DATASET=$1 # "imdb"
    RUN_NAME="ctrl-$MODEL_NAME-$DATASET"
    PROJECT="ctrl"
    BATCH_SIZE=$2
    MAX_SEQ_LEN=$3
    accelerate launch train_torch.py \
        --output_dir "./checkpoint/ctrl" \
        --project $PROJECT \
        --run_name $RUN_NAME \
        --do_eval --do_train \
        --model_name_or_path $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --task ctrl \
        --per_device_train_batch_size $BATCH_SIZE \
        --dataset_name $DATASET \
        --num_train_epochs 10 \
        --max_sequence_length $MAX_SEQ_LEN \
        --save_strategy epoch \
        --logging_steps 50
}

# train "imdb"
# train "emotion" 8 64
train "yelp_polarity" 8 256
# train "SetFit/bbc-news" 1 400
