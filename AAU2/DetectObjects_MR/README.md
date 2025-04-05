https://www.anaconda.com/download
https://developer.nvidia.com/cuda-downloads
conda create -n pytorch-env
conda activate pytorch-env
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
# TO RUN:
jupyter notebook