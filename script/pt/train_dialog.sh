export XRT_TPU_CONFIG="localservice;0;localhost:51011"
export USE_Torch=True


accelerate launch train_clm_hfacc.py config/dialog/gpt-j-base-rev3-step200k.yaml
accelerate launch train_clm_hfacc.py config/dialog/gpt-j-base-rev3-step400k.yaml
accelerate launch train_clm_hfacc.py config/dialog/gpt-j-base-rev3-master.yaml