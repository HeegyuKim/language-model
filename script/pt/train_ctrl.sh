DATASET="imdb"
MODEL_TYPE="causal-lm"
MODEL_NAME="gpt2-medium"
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
    --per_device_train_batch_size 4 \
    --dataset_name $DATASET \
    --num_train_epochs 10 \
    --max_sequence_length 128 \
    --save_strategy last \
    --logging_steps 50