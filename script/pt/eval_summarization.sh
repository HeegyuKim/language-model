export USE_TORCH=True

MODEL_TYPE="causal-lm"
PROJECT="nia-summ-gpt"

MODEL_NAME="kogpt-j-base"
RUN_NAME="$MODEL_NAME"

accelerate launch train_torch.py \
    --output_dir "./checkpoint/$RUN_NAME" \
    --project $PROJECT\
    --run_name $RUN_NAME \
    --do_eval --do_eval_generation \
    --model_name_or_path "checkpoint/nia-summ/$MODEL_NAME/checkpoint-epoch-17480-last/" \
    --model_type $MODEL_TYPE \
    --task nia-summ \
    --from_flax true \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps=8 \
    --max_sequence_length 1024 \
    --save_strategy last \
    --logging_steps 100
