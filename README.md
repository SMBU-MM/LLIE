# SynLLIE
This is a repository of two papers

Paper1: Enhancing Low-Light Images: A Synthetic Data Perspective on Practical and Generalizable Solutions [link](https://doi.org/10.1609/aaai.v39i6.32617)

Paper2: Towards Realistic Low-Light Image Enhancement via ISP-Driven Data Modeling

## 1. Create Environment


### 1.1 Install the environment with Pytorch 1.11

- Make Conda Environment
```
conda create -n synllie python=3.7
conda activate synllie
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

To download the pre-trained model, follow the link below：

Paper1: [Google Drive](https://drive.google.com/drive/folders/1eMRYNUgcTAduv4OVnajEkeELWaG_clr-?usp=drive_link) (There are three folders, including SNRNet, Retinexformer and RetinexMamba trained using our simulation method.)

Paper2: [Google Drive](https://drive.google.com/drive/folders/1tBYkZ7gXaI_sh8UGcXQfoPqAn4TB8fQU?usp=sharing)

Put them in folder `pretrained_models`

```shell
# activate the environment
conda activate synllie

# run (The path of the test set is modified in the configuration file.)
# test on LOL_v1
python3 Enhancement/test_from_dataset.py --opt Options/NewAttentionUNet2_test.yml --weights pretrained_models/AttentionUNet_LOL_v1.pth
# test on LOL_v2_real
python3 Enhancement/test_from_dataset.py --opt Options/NewAttentionUNet2_test.yml --weights pretrained_models/AttentionUNet_LOL_v2.pth
# test on other dataset
python3 Enhancement/test_from_dataset.py --opt Options/NewAttentionUNet2_test.yml --weights pretrained_models/AttentionUNet_baseline.pth 
