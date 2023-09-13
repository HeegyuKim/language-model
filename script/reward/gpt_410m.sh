export USE_TORCH=True

MODEL_TYPE="sequence-classification"
PROJECT="reward_model"
MODEL_OWNER="EleutherAI"
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
        --model_type $MODEL_TYPE \
        --task reward \
        --revision $MODEL_REVISION \
        --from_flax $FROM_FLAX \
        --num_train_epochs 3 \
        --num_labels 1 \
        --gradient_accumulation_steps 16 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --max_sequence_length 1024 \
        --save_strategy no \
        --logging_steps 1
}

MODEL_OWNER="EleutherAI"
train "pythia-160m-deduped"
train "pythia-410m-deduped"

# MODEL_OWNER="heegyu"
# train "WizardVicuna-Uncensored-pythia-160m-deduped"
# train "WizardVicuna-pythia-410m-deduped"