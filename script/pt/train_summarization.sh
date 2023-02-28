export USE_TORCH=True

MODEL_TYPE="causal-lm"
PROJECT="nia-summ-gpt"
MODEL_OWNER="heegyu"
MODEL_REVISION="main"
DTYPE="float32"

function train {
    MODEL_NAME="$MODEL_OWNER/$1"
    RUN_NAME="$1-$MODEL_REVISION"

    accelerate launch train_torch.py \
        --output_dir "./checkpoint/nia-summ" \
        --project $PROJECT\
        --run_name $RUN_NAME \
        --do_eval --do_train \
        --model_name_or_path $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --task nia-summ \
        --revision $MODEL_REVISION \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --max_sequence_length 1024 \
        --save_strategy last \
        --logging_steps 100

    rm -rf ~/.cache/huggingface/datasets
}

# train "ajoublue-gpt2-base"
# train "ajoublue-gpt2-base-24L"
# train "kogpt-j-base"
# train "kogpt-j-base-24L"
# DTYPE="bfloat16"
# train "ajoublue-gpt2-medium"
# train "kogpt-j-350m"

MODEL_OWNER="skt"
train "kogpt2-base-v2"