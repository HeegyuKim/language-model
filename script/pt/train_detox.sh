export USE_TORCH=True

PROJECT="toxic-seq-classification"

function train {
    MODEL_NAME="gpt2"
    RUN_NAME=$MODEL_NAME
    MODEL_TYPE="sequence-classification"
    TASK=toxic-token-classification
    TASK=toxic-sequence-classification

    accelerate launch train_torch.py \
        --output_dir "./checkpoint/$PROJECT" \
        --project $PROJECT\
        --run_name $RUN_NAME \
        --do_eval --do_train \
        --model_name_or_path $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --task $TASK \
        --num_train_epochs 5 \
        --num_labels 2 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --max_sequence_length 128 \
        --save_strategy epoch \
        --logging_steps 500
}

train