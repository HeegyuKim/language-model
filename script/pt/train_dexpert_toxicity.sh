export USE_TORCH=True

MODEL_TYPE="causal-lm"
PROJECT="dexpert"


function train {
    MODEL_NAME="gpt2"
    RUN_NAME="dexpert-$1"
    TASK=$1

    accelerate launch train_torch.py \
        --output_dir "./checkpoint/dexpert" \
        --project $PROJECT\
        --run_name $RUN_NAME \
        --do_eval --do_train \
        --model_name_or_path $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --task $TASK \
        --num_train_epochs 3 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --max_sequence_length 128 \
        --save_strategy last \
        --logging_steps 100
}

train "dexpert-toxic"
train "dexpert-non-toxic"