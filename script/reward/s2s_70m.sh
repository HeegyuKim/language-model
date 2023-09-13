export USE_TORCH=True

PROJECT="reward_model"
MODEL_OWNER="google"
MODEL_REVISION="main"
FROM_FLAX=false

function train {
    MODEL_NAME="$MODEL_OWNER/$1"
    RUN_NAME="rank-$1"

    accelerate launch train_torch.py \
        --output_dir "./checkpoint/$RUN_NAME" \
        --project $PROJECT\
        --run_name $RUN_NAME \
        --do_eval --do_train \
        --model_name_or_path $MODEL_NAME \
        --task seq2seq-rank-reward \
        --revision $MODEL_REVISION \
        --from_flax $FROM_FLAX \
        --num_train_epochs 3 \
        --num_labels 1 \
        --max_train_samples 1000 \
        --max_eval_samples 1000 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --max_sequence_length 512 \
        --decoder_max_sequence_length 128 \
        --save_strategy no \
        --logging_steps 32
}

train "flan-t5-small"
