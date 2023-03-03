export USE_TORCH=True

MODEL_TYPE="causal-lm"
PROJECT="nia-dialog"

eval_dialog() {
    MODEL_NAME=$1
    RUN_NAME="$MODEL_NAME"

    accelerate launch train_torch.py \
        --output_dir "./checkpoint/$RUN_NAME" \
        --project $PROJECT\
        --run_name $RUN_NAME \
        --do_eval --do_eval_generate \
        --model_name_or_path "checkpoint/dialog/$MODEL_NAME/" \
        --model_type $MODEL_TYPE \
        --task nia-dialog \
        --from_flax true \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps=8 \
        --max_sequence_length 1024 \
        --save_strategy no \
        --logging_steps 100
}

# eval_dialog "kogpt-j-base-rev4-dialog"

PROJECT="nia-summ-gpt"

eval_sum() {
    MODEL_NAME=$1
    RUN_NAME="$MODEL_NAME"
    accelerate launch train_torch.py \
        --output_dir "./checkpoint/$RUN_NAME" \
        --project $PROJECT\
        --run_name $RUN_NAME \
        --do_eval --do_eval_generate \
        --model_name_or_path "checkpoint/nia-summ/$MODEL_NAME" \
        --model_type $MODEL_TYPE \
        --task nia-summ \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps=8 \
        --max_sequence_length 1024 \
        --save_strategy no \
        --logging_steps 100

}

# eval_sum "kogpt-j-base-nia-summ"
eval_sum "skt-nia-summ"