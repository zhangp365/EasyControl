# Implementation of EasyControl

EasyControl: Adding Efficient and Flexible Control for Diffusion Transformer

<a href='https://arxiv.org/pdf/2503.07027'><img src='https://img.shields.io/badge/Technique-Report-red'></a> 

> *[Yuxuan Zhang](https://xiaojiu-z.github.io/YuxuanZhang.github.io/), [Yirui Yuan](https://github.com/Reynoldyy), [Yiren Song](https://scholar.google.com.hk/citations?user=L2YS0jgAAAAJ), [Haofan Wang](https://haofanwang.github.io/), [Jiaming Liu](https://scholar.google.com/citations?user=SmL7oMQAAAAJ&hl=en)*
> <br>
> Tiamat AI, ShanghaiTech University, National University of Singapore, Liblib AI

<img src='assets/teaser.jpg'>

## Features
* **Motivation:**: The architecture of diffusion models is transitioning from Unet-based to DiT (Diffusion Transformer). However, the DiT ecosystem lacks mature plugin support and faces challenges such as efficiency bottlenecks, conflicts in multi-condition coordination, and insufficient model adaptability, particularly in zero-shot multi-condition scenarios where these issues are most pronounced.
* **Contribution:**: We propose Easy Control, an efficient and flexible unified conditional guidance DiT framework. By incorporating a lightweight Condition Injection LoRA module, a Position-Aware Training Paradigm, and a combination of causal attention mechanisms with KV Cache technology, we significantly enhance model compatibility, generation flexibility, and inference efficiency.
<img src='assets/method.jpg'>

## News
- **2025-03-12**: ⭐️ Inference code are released. Once we have ensured that everything is functioning correctly, the new model will be merged into this repository. Stay tuned for updates! 😊.

## Todo List
1. - [x] Inference code 
2. - [ ] Pre-trained weights 
3. - [ ] Training code


## Citation
```
@misc{zhang2025easycontroladdingefficientflexible,
      title={EasyControl: Adding Efficient and Flexible Control for Diffusion Transformer}, 
      author={Yuxuan Zhang and Yirui Yuan and Yiren Song and Haofan Wang and Jiaming Liu},
      year={2025},
      eprint={2503.07027},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.07027}, 
}
```
