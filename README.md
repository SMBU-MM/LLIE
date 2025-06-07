# SynLLIE-NTIRE2025

## 1. Create Environment


### 1.1 Install the environment with Pytorch 1.11

- Make Conda Environment
```
conda create -n ntire python=3.7
conda activate ntire
```

- Install Dependencies
```
conda install pytorch=1.11 torchvision cudatoolkit=11.3 -c pytorch

pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard

pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips
```

- Install BasicSR
```
python setup.py develop --no_cuda_ext
```

## 2. Testing

Download our models from [Google Drive](https://drive.google.com/drive/folders/1utvLBrYmZODgsMVohaJzVNOTbiK-KQHK?usp=drive_link). Put them in folder `pretrain_models`

```shell
# activate the environment
conda activate ntire

# run
python3 Enhancement/test_from_dataset.py --opt Options/NTIRE_LLIE2025.yml --weights pretrain_models/NTIRE_LLIE2025.pth 
