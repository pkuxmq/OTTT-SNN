# OTTT-SNN
This is the PyTorch implementation of paper: Online Training Through Time for Spiking Neural Networks **(NeurIPS 2022)**. \[[arxiv](https://arxiv.org/abs/2210.04195)\].

## Dependencies and Installation
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch, torchvision](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install numpy opencv-python`

## Training
For OTTT$_A$, run as following:

	python train_cifar.py -data_dir path_to_data_dir -dataset cifar10 -out_dir log_checkpoint_name -gpu-id 0

	# For VGG-F model
	python train_cifar.py -data_dir path_to_data_dir -dataset cifar100 -out_dir log_checkpoint_name -gpu-id 0 -model online_spiking_vgg11f_ws

	python train_cifar10dvs.py -data_dir path_to_data_dir -out_dir log_checkpoint_name -gpu-id 0

	python train_imagenet.py -data_dir path_to_data_dir -out_dir log_checkpoint_name -gpu-id 0

For OTTT$_O$, add the argument -online\_update as:

	python train_cifar.py -data_dir path_to_data_dir -dataset cifar10 -out_dir log_checkpoint_name -gpu-id 0 -online_update

The default hyperparameters in the code are the same as in the paper.

Note: Current codes only support single-gpu training.

## Testing
We provide the example code to calculate the firing rate statistics during evaluation. Run as following:

	python get_rate_cifar.py -data_dir path_to_data_dir -dataset cifar10 -gpu-id 0 -resume path_to_checkpoint

	python get_rate_imagenet.py -data_dir path_to_data_dir -gpu-id 0 -resume path_to_checkpoint

Some pretrained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1eDn3mVgfBHTLBfb--WawgA5Qms4oFbZ4?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1q0ljZiCVIUW41Hh-aol2Zg) (extraction code: gppq).

## Acknowledgement

Some codes for the neuron model and data prepoccessing are adapted from the [spikingjelly](https://github.com/fangwei123456/spikingjelly) repository, and the codes for some utils are from the [pytorch-classification](https://github.com/bearpaw/pytorch-classification) repository.

## Contact
If you have any questions, please contact <mingqing_xiao@pku.edu.cn>.
