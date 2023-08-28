## Learnable Nonlocal Self-similarity of Deep Features for Image Denoising





Training on AWGN
----------
- Training Set:
DIV2K(https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) + Flickr2K(https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) +
BSD500(http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz) + WED(http://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar)


- All the pretrained models are put in `./model_zoo`.




Testing on AWGN
----------
- Testing Set:
Grayscale: Set12 
  (https://github.com/cszn/FFDNet/tree/master/testsets)
```bash
python main_test.py --task gray_dn --noise 15 --model_path model_zoo/model.pth --folder_gt testsets/set12
```




References
----------
```BibTex
@inproceedings{liang2021swinir,
title={SwinIR: Image Restoration Using Swin Transformer},
author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
booktitle={IEEE International Conference on Computer Vision Workshops},
pages={1833--1844},
year={2021}
}
```
