DATASET="imdb"
DATASET="heegyu/news-category-balanced-top10"
MODEL_TYPE="causal-lm"
MODEL_NAME="gpt2"
RUN_NAME="ctrl-$MODEL_NAME-$DATASET"
PROJECT="ctrl"

accelerate launch train_torch.py \
    --output_dir "./checkpoint/$RUN_NAME" \
    --project $PROJECT \
    --run_name $RUN_NAME \
    --do_eval --do_train \
    --model_name_or_path $MODEL_NAME \
    --model_type $MODEL_TYPE \
    --task ctrl \
    --per_device_train_batch_size 8 \
    --dataset_name $DATASET \
    --num_train_epochs 5 \
    --max_sequence_length 256 \
    --save_strategy last \
    --logging_steps 50