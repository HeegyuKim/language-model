export USE_TORCH=True

# MODEL_NAME="heegyu/kogpt-j-base"
MODEL_TYPE="sequence-classification"
# RUN_NAME="nsmc-kogpt-j-base"
PROJECT="nsmc"

# --do_train \
function train_nsmc {
    MODEL_NAME="heegyu/$1"
    RUN_NAME="nsmc-$1"

    accelerate launch train_classifier.py \
        --output_dir "./checkpoint/$RUN_NAME" \
        --project $PROJECT\
        --run_name $RUN_NAME \
        --do_eval --do_train \
        --num_procs 8 \
        --model_name_or_path $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --dataset_name nsmc \
        --num_train_epochs 3 \
        --num_labels 2 \
        --from_flax true \
        --max_sequence_length 128 \
        --save_strategy no \
        --logging_steps 100
}

# train_nsmc "ajoublue-gpt2-base"
# train_nsmc "ajoublue-gpt2-base-24L"

# train_nsmc "kogpt-j-base"
# train_nsmc "kogpt-j-base-24L"
# train_nsmc "kogpt-j-350m"


function train_nsmc2 {
    MODEL_NAME="$2/$1"
    RUN_NAME="nsmc-$1"

    # accelerate launch train_classifier.py \
    accelerate launch train_torch.py \
        --output_dir "./checkpoint/$RUN_NAME" \
        --project $PROJECT\
        --run_name $RUN_NAME \
        --do_eval --do_train \
        --num_procs 8 \
        --model_name_or_path $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --task nsmc \
        --num_train_epochs 3 \
        --num_labels 2 \
        --max_sequence_length 128 \
        --save_strategy no \
        --logging_steps 100
}

train_nsmc2 "kogpt2-base-v2" "skt"