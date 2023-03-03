export USE_TORCH=True

MODEL_TYPE="sequence-classification"
PROJECT="news-category-classification"

function train {
    MODEL_NAME=$1
    RUN_NAME=$1

    accelerate launch train_torch.py \
        --output_dir "./checkpoint/$PROJECT" \
        --project $PROJECT \
        --run_name $RUN_NAME \
        --do_eval --do_train \
        --model_name_or_path $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --task news-category-top10 \
        --num_train_epochs 5 \
        --num_labels 10 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --max_sequence_length 64 \
        --save_strategy epoch \
        --logging_steps 100
}

train "roberta-base"