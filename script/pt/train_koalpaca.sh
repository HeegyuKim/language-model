export USE_TORCH=True

MODEL_TYPE="causal-lm"
PROJECT="koalpaca"
MODEL_OWNER="heegyu"
MODEL_REVISION="main"

function train {
    MODEL_NAME="$MODEL_OWNER/$1"
    RUN_NAME="$1"

    accelerate launch train_torch.py \
        --output_dir "./checkpoint/$PROJECT" \
        --project $PROJECT\
        --run_name $RUN_NAME \
        --do_eval --do_train \
        --model_name_or_path $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --task koalpaca \
        --num_train_epochs 10 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --max_sequence_length 256 \
        --save_strategy last \
        --logging_steps 250
}

train "ajoublue-gpt2-medium"