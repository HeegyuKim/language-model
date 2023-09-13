conda create -n torch38 python=3.8 -y
conda activate torch38
rm ~/.local/bin/accelerate

pip install torch~=2.0.0 https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-2.0-cp38-cp38-linux_x86_64.whl
pip install -r requirements.txt