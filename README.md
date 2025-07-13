<div align="center">

# Unsupervised Quality Control and Enhancement of Polyp Segmentation in Colonoscopy Videos using Spatiotemporal Consistency

Yujia Li<sup>1</sup>, Tao Zhou<sup>1</sup>, Ruixuan Wang<sup>2</sup>, Shuo Wang<sup>3</sup>, Yizhe Zhang<sup>1</sup>

<sup>1</sup> Nanjing University of Science and Technology, China

<sup>2</sup> Sun Yat-sen University, China

<sup>3</sup> Fudan University, China

</div>

## Getting Started

#### Framework Clone

```
git clone https://github.com/LYJ-NJUST/Seg-Quality-Control.git
cd Seg-Quality-Control
```

#### SAM2 Installation 

SAM 2 needs to be installed into the framework directory first before use. The code requires `python>=3.10`, as well as `torch>=2.3.1` and `torchvision>=0.18.1`. Please follow the instructions [here](https://github.com/facebookresearch/sam2?tab=readme-ov-file) to install SAM2 and download checkpoints.

Please see [INSTALL.md](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md) from the original SAM 2 repository for FAQs on potential issues and solutions.

Install other requirements:
```
pip install opencv-python scipy shutil json gc
```

#### Data Preparation

Please place the frame sequence and the original segmentation masks from the polyp segmentation model into the `frames` folder and `origin_masks` folder under the `example` directory, respectively.

#### Exanple
```
python example.py 
```

## Citation

Please consider citing our paper and the wonderful `SAM 2` if you found our work interesting and useful.
```
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}

@todo
```
