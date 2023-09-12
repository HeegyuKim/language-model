export USE_TORCH=True

MODEL_TYPE="causal-lm"
PROJECT="koalpaca"
MODEL_OWNER="heegyu"
MODEL_REVISION="main"

function train {
    MODEL_NAME="$MODEL_OWNER/$1"
    RUN_NAME="$1"

    accelerate launch \
        --config_file ./config/xla_fsdp.yaml \
        train_torch.py \
        --output_dir "./checkpoint/$PROJECT" \
        --project $PROJECT\
        --run_name $RUN_NAME \
        --do_eval --do_train \
        --model_name_or_path $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --task koalpaca \
        --num_train_epochs 5 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --max_sequence_length 256 \
        --save_strategy last \
        --logging_steps 250
}

train "ajoublue-gpt2-base"
# MODEL_OWNER="EleutherAI"
# train "polyglot-ko-1.3b"