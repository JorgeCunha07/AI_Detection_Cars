https://www.anaconda.com/download
https://developer.nvidia.com/cuda-downloads
conda create -n pytorch-env
conda activate pytorch-env
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
python -m ipykernel install --user --name=torch_gpu2 --display-name "Python [conda env:torch_gpu2]" 
# TO RUN:
jupyter notebook