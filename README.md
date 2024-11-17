# Vidu4D

This repository is an implementation of "Vidu4D: Single Generated Video to High-Fidelity 4D Reconstruction with Dynamic Gaussian Surfels", in NeurIPS 2024.

> [**Vidu4D: Single Generated Video to High-Fidelity 4D Reconstruction with Dynamic Gaussian Surfels**](https://arxiv.org/abs/2405.16822)<br>
> [Yikai Wang](https://yikaiw.github.io/)\*, [Xinzhou Wang](https://zz7379.github.io/)\*, [Zilong Chen](https://scholar.google.com/citations?user=2pbka1gAAAAJ&hl=en), [Zhengyi Wang](https://thuwzy.github.io/), [Fuchun Sun](https://scholar.google.com/citations?user=DbviELoAAAAJ&hl=en), [Jun Zhu](https://ml.cs.tsinghua.edu.cn/~jun/index.shtml)

<div align="left">
  <a href="https://vidu4d-dgs.github.io/"><img src="https://img.shields.io/badge/🌐-Website%20-blue.svg?label=Project"></a> &ensp;
  <a href="https://arxiv.org/abs/2405.16822"><img src="https://img.shields.io/static/v1?label=Paper&message=arXiv&color=red"></a> &ensp;
</div>

## Overview
<p align="center">
    <img src="assets/method.png" style="width: 90%;">
</p>

## 4D Results
See accompanying videos in our [project page](https://vidu4d-dgs.github.io/) for better visual quality.
<p align="center">
    <img src="assets/results.png" style="width: 90%;">
</p>

## Training
First, specify the experiment name, 
```
seqname=cheetah
```
Then, create a ```seqname``` folder under ```database/raw``` and put a video (e.g., ```0.mp4```) in it.

Stage 1, data preprocess, including data cropping, optical flow, DINO feature, rough poses, etc,
```
python scripts/run_preprocess.py $seqname $seqname other "0"
```

Stage 2, optimize neural sdf,
```
python lab4d/train.py --seqname $seqname --logname base --fg_motion bob --num_rounds 21 --rgb_timefree --rgb_dirfree
```


Stage 3, optimize Gaussian surfels,
```
python lab4d/train.py --seqname ${seqname} --logname gs-frzwarp --fg_motion gs-bob --num_rounds 61 --load_path logdir/${seqname}-base/ckpt_0020.pth --gs_init_mesh logdir/${seqname}-base/021-fg-geo.obj --imgs_per_gpu 1 --pixels_per_image -1 --eval_res 256 --rgb_timefree --rgb_dirfree --rgb_loss_only --gs_optim_warp=False --data_prefix full --force_center_cam
```

Render after training,
```
python lab4d/render.py --flagfile=logdir/${seqname}-gs-frzwarp/opts.log --load_suffix latest --render_res 512
```

## Todo
- DGS Refinement
- Benchmarks

## Acknowledgements 
Our code is built based on [2DGS](https://github.com/hbb1/2d-gaussian-splatting) and [Lab4D](https://github.com/lab4d-org/lab4d). We thank the authors for their great repos.

## BibTeX
```
@inproceedings{wang2024vidu4d,
  title={Vidu4D: Single Generated Video to High-Fidelity 4D Reconstruction with Dynamic Gaussian Surfels},
  author={Yikai Wang and Xinzhou Wang and Zilong Chen and Zhengyi Wang and Fuchun Sun and Jun Zhu},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```
