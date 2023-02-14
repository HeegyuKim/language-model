export USE_TORCH=True

MODEL_TYPE="sequence-classification"
PROJECT="nsmc"
MODEL_OWNER="heegyu"
MODEL_REVISION="main"
FROM_FLAX=false

function train_nsmc {
    MODEL_NAME="$MODEL_OWNER/$1"
    RUN_NAME="nsmc-$1"

    accelerate launch train_torch.py \
        --output_dir "./checkpoint/$RUN_NAME" \
        --project $PROJECT\
        --run_name $RUN_NAME \
        --do_eval --do_train \
        --model_name_or_path $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --task nsmc \
        --revision $MODEL_REVISION \
        --from_flax $FROM_FLAX \
        --num_train_epochs 3 \
        --num_labels 2 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --max_sequence_length 128 \
        --save_strategy no \
        --logging_steps 100
}

# train_nsmc "ajoublue-gpt2-base"
train_nsmc "ajoublue-gpt2-base-24L"
train_nsmc "ajoublue-gpt2-medium"

MODEL_REVISION="master"
FROM_FLAX=true

train_nsmc "kogpt-j-base"

MODEL_REVISION="main"
FROM_FLAX=false

train_nsmc "kogpt-j-base-24L"
train_nsmc "kogpt-j-350m"

MODEL_OWNER="skt"
train_nsmc "kogpt2-base-v2"