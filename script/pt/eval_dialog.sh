export USE_TORCH=True

MODEL_TYPE="causal-lm"
PROJECT="nia-summ-gpt"

train() {
    MODEL_NAME=$1
    RUN_NAME="$MODEL_NAME"

    accelerate launch train_torch.py \
        --output_dir "./checkpoint/$RUN_NAME" \
        --project $PROJECT\
        --run_name $RUN_NAME \
        --do_eval --do_eval_generate \
        --model_name_or_path "checkpoint/dialog/$MODEL_NAME/checkpoint-epoch-17480-last/" \
        --model_type $MODEL_TYPE \
        --task nia-dialog \
        --from_flax true \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps=8 \
        --max_sequence_length 1024 \
        --save_strategy last \
        --logging_steps 100
}

train "ajoublue-gpt2-base"
train "ajoublue-gpt2-base-24L"
train "ajoublue-gpt2-medium"

train "kogpt-j-base"
train "kogpt-j-base-24L"
train "kogpt-j-350m"