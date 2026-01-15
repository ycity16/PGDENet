## PGDENet
Abstract:
Real-world video degradation shows pronounced spatial non-uniformity and
temporal dynamics. Conventional unified models struggle with time-varying
unknown degradations (TUD), often causing inaccurate frame alignment, insuf-
ficient degradation adaptation, and temporal inconsistencies. To address these
challenges, we propose Prompt-Guided Dynamic Expert Network (PGDENet),
a prompt-driven All-in-One Video Restoration network. By integrating prompt
learning and a mixture-of-experts mechanism into a recurrent propagation
paradigm, PGDENet achieves adaptive and efficient restoration of complex TUD.
To reduce alignment errors, we design the Prompt-Guided Deformable Align-
ment (PGDA) module, which uses content-adaptive dynamic prompts to jointly
constrain alignment and modulate features. For diverse degradations, we intro-
duce the Mixture-of-Dimensions Experts (MoDE) system, employing dual-branch
(spatial and high-frequency) routing to sparsely activate optimal sub-networks
from a heterogeneous expert pool, balancing efficiency and performance. We
further propose the Dynamic Prompt Expert Modulator (DPEM), which gen-
erates input-adaptive modulation signals from a learnable visual prompt pool,
enhancing semantic representation while providing reliable degradation-aware
priors for expert routing. A key-frame guided strategy is also incorporated to
maintain robust global temporal consistency in long sequences. Extensive exper-
iments validate PGDENet’s effectiveness on TUD restoration. Comprehensive
evaluations were performed on two synthetic datasets, each featuring seven degra-
dation types with randomly varying corruption levels

## Network Architecture
![PGDENet.jpg](PGDENet.jpg)


## Installation

Ubuntu18.04、CUDA11.1、CUDNN8、Python3.7.5、OpenCV4.5.1、PyTorch1.12

1. `pip install -r requirements.txt`
2. `pip install -U openmim` 
3. `mim install mmcv`


## Dataset Preparation

### Training Data

1. Download the DAVIS dataset from (...To be uploaded).
2. Synthesize the low-quality (LQ) videos through `scripts/data_preparation/synthesize_datasets.py`.
```
python scripts/data_preparation/synthesize_datasets.py --input_dir 'The root of DAVIS' --output_dir 'LQ roots' --continuous_frames 6
```
3. Generate meta_info files for the training sets.
> This step can be ommited if you use the `DAVIS` dataset for training since the `DAVIS_meta_info.txt` file is already generated. (located in `basicsr/data/meta_info/DAVIS_meta_info.txt`)
```
python scripts/data_preparation/generate_meta_info.py --dataset_path 'The root of training sets'
# The meta infomation file is automatically saved in `basicsr/data/meta_info/training_meta_info.txt`
```

### Testing Data

The test sets can be downloaded from (...To be uploaded) or you can synthesize them through `synthesize_datasets.py`.

## Testing

1. Download the pretrained weights of SPyNet and PGENet from (...To be uploaded)(通过网盘分享的文件：models
链接: https://pan.baidu.com/s/1d_O5nnns4WnPcCcLi328ig?pwd=24qj 提取码: 24qj 
--来自百度网盘超级会员v6的分享).
2. Put the SPyNet weights to `experiments/pretrained_models/flownet/` and PGENet weights to `experiments/pretrained_models/`.
3. Modify the option yaml file in `options/test/` to begin. Then run the testing.
```
python basicsr/test.py -opt options/test/test_DAVIS_T6.yml
```

## Training

1. Put the SPyNet weights to `experiments/pretrained_models/flownet/`.
2. Modify the option yaml file in `options/train/` to begin. Then run training.
```
python basicsr/train.py -opt options/train/train_PGENet_DAVIS.yml
```

## Acknowledgements
The codes are based on [BasicSR](https://github.com/XPixelGroup/BasicSR).The data synthesis code is from [AverNet](https://github.com/XLearning-SCU/2024-NeurIPS-AverNet.). Thanks the authors for their codes!
