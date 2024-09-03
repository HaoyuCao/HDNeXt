# HDNeXt
This repository is the official implementation of our work, HDNeXt: Hybrid Dynamic MedNeXt with Level Set Regularization for Medical Image Segmentation. In this repository, we implement the 2D version of MedNeXt and propose our improved HDNeXt network based on this version.

The repository will mainly implement the following modules:

1. MedNeXt 2D module and network
2. Dynamic MedNeXt module
3. Curvature-Region Loss based on the level set method --- one simple implementation version

Pretrained weights will be released soon...

Codes copied a lot from transunet, MedNeXt, ConvNeXt V1 / V2, swin-transformer, etc. Thanks for their great works!

## Requirements
- Python 3.8
- PyTorch 1.10.0
- CUDA 11.3
- torchvision 0.11.1
- timm 0.4.12
- einops 0.3.0
- scikit-image 0.18.1
- scipy 1.7.1
- numpy 1.21.2
- tqdm 4.62.3
- albumentations 0.4.3
- matplotlib 3.4.3
- scipy 1.7.1
- pandas 1.3.3